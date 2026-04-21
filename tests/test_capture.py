"""Unit tests for the pursuit-capture helper (pure logic only, no CARLA)."""

import pytest

from skycop.cv.capture import weather_preset


def test_weather_preset_returns_known_preset():
    # This runs against the real carla module imported inside capture.py —
    # ClearNoon exists across all 0.9.x versions.
    import carla
    result = weather_preset("ClearNoon")
    assert isinstance(result, carla.WeatherParameters)


def test_weather_preset_rejects_unknown_name_with_helpful_message():
    with pytest.raises(ValueError) as ei:
        weather_preset("NoSuchWeather")
    msg = str(ei.value)
    assert "NoSuchWeather" in msg
    assert "Valid:" in msg
