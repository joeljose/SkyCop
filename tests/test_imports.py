"""Smoke test — the skycop package and its submodules import cleanly.

Runs without a live CARLA server (we don't call into it here, we just import).
"""


def test_package_imports():
    import skycop
    assert skycop.__version__


def test_sim_imports():
    from skycop.sim import carla_image_to_bgr, connect, spawn_aerial_camera, synchronous_mode
    assert callable(connect)
    assert callable(synchronous_mode)
    assert callable(spawn_aerial_camera)
    assert callable(carla_image_to_bgr)


def test_dashboard_imports():
    from skycop.dashboard import MJPEGServer
    server = MJPEGServer(title="test")
    assert server.app is not None


def test_cv_control_stubs_import():
    import skycop.control  # noqa: F401
    import skycop.cv  # noqa: F401
