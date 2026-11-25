"""Tests du délai de double-clic du système."""

import sys

from airdesk.actuator.system import (
    DEFAULT_DOUBLE_CLICK_MS,
    resolve_double_click_ms,
    system_double_click_ms,
)


def test_system_double_click_time_is_plausible_or_unknown():
    value = system_double_click_ms()
    if sys.platform == "win32":
        assert 100 <= value <= 5000
    else:
        assert value is None


def test_resolve_auto_and_numbers(monkeypatch):
    import airdesk.actuator.system as system

    monkeypatch.setattr(system, "system_double_click_ms", lambda: None)
    assert resolve_double_click_ms("auto") == DEFAULT_DOUBLE_CLICK_MS
    monkeypatch.setattr(system, "system_double_click_ms", lambda: 650)
    assert resolve_double_click_ms("auto") == 650
    assert resolve_double_click_ms(0) == 0
    assert resolve_double_click_ms(300) == 300
