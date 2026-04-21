"""Unit tests for skycop.cv.tracking_eval — suspect-continuity logic (pure, no CARLA)."""

from skycop.cv.tracking_eval import GTBox, TrackerBox, evaluate_suspect_continuity


def _gt(aid: int, box=(0, 0, 20, 20), vis=1.0) -> GTBox:
    return GTBox(actor_id=aid, x1=box[0], y1=box[1], x2=box[2], y2=box[3], visibility=vis)


def _tb(tid, box=(0, 0, 20, 20), conf=0.9) -> TrackerBox:
    return TrackerBox(track_id=tid, x1=box[0], y1=box[1], x2=box[2], y2=box[3], confidence=conf)


def test_perfect_continuity_returns_1_0():
    # 5 frames, suspect present and detected each frame with stable track_id=7
    gt_frames = [[_gt(aid=100)] for _ in range(5)]
    tracker_frames = [[_tb(tid=7)] for _ in range(5)]

    result = evaluate_suspect_continuity(tracker_frames, gt_frames, suspect_actor_id=100)
    assert result.continuity == 1.0
    assert result.initial_lock_track_id == 7
    assert result.id_switches == 0


def test_one_id_switch_lowers_continuity():
    # 5 frames: id=7 first 3 frames, id=8 last 2. Continuity = 3/5 = 0.6
    gt_frames = [[_gt(aid=100)] for _ in range(5)]
    tracker_frames = [
        [_tb(tid=7)], [_tb(tid=7)], [_tb(tid=7)],
        [_tb(tid=8)], [_tb(tid=8)],
    ]
    result = evaluate_suspect_continuity(tracker_frames, gt_frames, suspect_actor_id=100)
    assert 0.55 < result.continuity < 0.65
    assert result.id_switches == 2
    assert result.initial_lock_track_id == 7


def test_suspect_absent_frames_dont_count_against():
    # 5 frames, suspect absent in middle 2 frames. Tracker is perfect on remaining 3.
    gt_frames = [
        [_gt(aid=100)],
        [_gt(aid=100)],
        [],                # suspect absent
        [],                # suspect absent
        [_gt(aid=100)],
    ]
    tracker_frames = [
        [_tb(tid=5)], [_tb(tid=5)],
        [],              # no detection
        [],              # no detection
        [_tb(tid=5)],
    ]
    result = evaluate_suspect_continuity(tracker_frames, gt_frames, suspect_actor_id=100)
    assert result.continuity == 1.0
    assert result.n_frames == 5
    assert result.n_frames_suspect_present == 3
    assert result.n_frames_suspect_detected == 3


def test_non_suspect_tracks_ignored():
    # 3 frames, multiple vehicles. Suspect (aid=100) vs NPC (aid=200).
    # Tracker is messy on NPCs but perfect on suspect.
    gt_frames = [
        [_gt(aid=100, box=(0, 0, 20, 20)), _gt(aid=200, box=(50, 50, 70, 70))],
        [_gt(aid=100, box=(0, 0, 20, 20)), _gt(aid=200, box=(50, 50, 70, 70))],
        [_gt(aid=100, box=(0, 0, 20, 20)), _gt(aid=200, box=(50, 50, 70, 70))],
    ]
    tracker_frames = [
        [_tb(tid=3, box=(0, 0, 20, 20)), _tb(tid=9, box=(50, 50, 70, 70))],
        [_tb(tid=3, box=(0, 0, 20, 20)), _tb(tid=10, box=(50, 50, 70, 70))],  # npc id switched
        [_tb(tid=3, box=(0, 0, 20, 20)), _tb(tid=11, box=(50, 50, 70, 70))],  # again
    ]
    result = evaluate_suspect_continuity(tracker_frames, gt_frames, suspect_actor_id=100)
    assert result.continuity == 1.0  # suspect never switched
    assert result.id_switches == 0


def test_reacquisition_gap_after_missed_detection():
    # Suspect detected in frames 0,1 with id=5. Missed in frames 2,3 (no tracker output).
    # Re-acquired in frame 4 with same id=5.
    gt_frames = [[_gt(aid=100)] for _ in range(5)]
    tracker_frames = [
        [_tb(tid=5)],    # locked
        [_tb(tid=5)],
        [],              # missed
        [],              # missed
        [_tb(tid=5)],    # reacquired
    ]
    result = evaluate_suspect_continuity(tracker_frames, gt_frames, suspect_actor_id=100)
    assert result.continuity == 1.0  # all detection-present frames consistent
    # One re-acquisition event; gap ended at frame 4 from a missed run starting at frame 2 → gap=2
    assert len(result.reacquisition_frames) == 1
    assert result.reacquisition_frames[0] == 2


def test_no_suspect_at_all():
    # No suspect in any frame; edge case. Continuity 0 by default.
    gt_frames = [[_gt(aid=200)] for _ in range(3)]
    tracker_frames = [[_tb(tid=1)] for _ in range(3)]
    result = evaluate_suspect_continuity(tracker_frames, gt_frames, suspect_actor_id=100)
    assert result.continuity == 0.0
    assert result.n_frames_suspect_present == 0
    assert result.initial_lock_track_id is None
