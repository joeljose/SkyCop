"""mAP scoring on a YOLO-format eval set.

Uses torchmetrics' MeanAveragePrecision for explicit control over thresholds
and per-class breakdown. Runs entirely on CPU — eval sets are small.

`class_filter` restricts scoring to a subset of SkyCop classes. Used for
the pretrained baseline, which has no COCO-equivalent for our `van` class —
scoring pretrained on 4 classes would unfairly always-miss van and inflate
the fine-tune's apparent improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EvalBox:
    class_idx: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0      # 1.0 for ground truth


def read_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[EvalBox]:
    """Read a YOLO-format label file, return pixel-coordinate boxes.

    YOLO file format: one line per box, `cls cx cy w h` all normalized to [0,1].
    """
    if not label_path.exists():
        return []
    out: list[EvalBox] = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        cls_idx = int(parts[0])
        cx, cy, w, h = (float(x) for x in parts[1:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        out.append(EvalBox(class_idx=cls_idx, x1=x1, y1=y1, x2=x2, y2=y2))
    return out


def filter_classes(
    boxes: list[EvalBox], keep: set[int] | None
) -> list[EvalBox]:
    """Drop boxes whose class_idx is not in `keep`. None = keep all."""
    if keep is None:
        return boxes
    return [b for b in boxes if b.class_idx in keep]


@dataclass
class EvalMetrics:
    map_50: float                         # mAP at IoU=0.5
    map_50_95: float                      # mAP averaged over IoU 0.5..0.95
    per_class_ap_50: dict[int, float]     # class_idx → AP@0.5
    n_frames: int
    n_predictions: int
    n_ground_truths: int
    class_filter: list[int] | None


def score_predictions(
    preds_per_frame: list[list[EvalBox]],
    gts_per_frame: list[list[EvalBox]],
    class_filter: set[int] | None = None,
) -> EvalMetrics:
    """Compute mAP using torchmetrics.detection.MeanAveragePrecision.

    `preds_per_frame[i]` and `gts_per_frame[i]` correspond to the same image.
    Boxes are in `xyxy` pixel coordinates.
    """
    import torch
    from torchmetrics.detection import MeanAveragePrecision

    preds = [filter_classes(p, class_filter) for p in preds_per_frame]
    gts = [filter_classes(g, class_filter) for g in gts_per_frame]

    if class_filter is not None:
        # torchmetrics handles arbitrary labels, but we sort for stable per_class output
        pass

    tm_preds = []
    for frame_preds in preds:
        if frame_preds:
            tm_preds.append({
                "boxes": torch.tensor([[b.x1, b.y1, b.x2, b.y2] for b in frame_preds]),
                "scores": torch.tensor([b.confidence for b in frame_preds]),
                "labels": torch.tensor([b.class_idx for b in frame_preds]),
            })
        else:
            tm_preds.append({
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.int64),
            })

    tm_gts = []
    for frame_gts in gts:
        if frame_gts:
            tm_gts.append({
                "boxes": torch.tensor([[b.x1, b.y1, b.x2, b.y2] for b in frame_gts]),
                "labels": torch.tensor([b.class_idx for b in frame_gts]),
            })
        else:
            tm_gts.append({
                "boxes": torch.zeros((0, 4)),
                "labels": torch.zeros((0,), dtype=torch.int64),
            })

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(tm_preds, tm_gts)
    result = metric.compute()

    per_class_ap_50: dict[int, float] = {}
    map_50_per_class = result.get("map_50_per_class")
    classes_tensor = result.get("classes")
    if map_50_per_class is not None and classes_tensor is not None:
        for cls, ap in zip(classes_tensor.tolist(), map_50_per_class.tolist(), strict=True):
            if not np.isnan(ap) and ap >= 0:
                per_class_ap_50[int(cls)] = float(ap)

    return EvalMetrics(
        map_50=float(result["map_50"]),
        map_50_95=float(result["map"]),
        per_class_ap_50=per_class_ap_50,
        n_frames=len(preds),
        n_predictions=sum(len(p) for p in preds),
        n_ground_truths=sum(len(g) for g in gts),
        class_filter=sorted(class_filter) if class_filter is not None else None,
    )
