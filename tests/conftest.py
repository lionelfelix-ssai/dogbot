import pytest

try:
    import hailo_platform  # noqa: F401
    HAILO = True
except Exception:
    HAILO = False


def pytest_collection_modifyitems(config, items):
    if HAILO:
        return
    skip = pytest.mark.skip(reason="hailo_platform not available")
    for item in items:
        if "pi" in item.keywords:
            item.add_marker(skip)
