"""Tests for learned room-to-room transition model."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from custom_components.area_occupancy.data.transition_model import (
    collect_transition_increments,
    weights_for_adjacent_targets,
)


def test_weights_uniform_without_counts() -> None:
    """No data → equal split across adjacents."""
    w = weights_for_adjacent_targets(["Hall", "Kitchen"], {})
    assert pytest.approx(sum(w.values()), rel=1e-6) == 1.0
    assert w["Hall"] == pytest.approx(0.5)
    assert w["Kitchen"] == pytest.approx(0.5)


def test_weights_follow_counts() -> None:
    """Higher count → higher weight (Laplace-smoothed)."""
    w = weights_for_adjacent_targets(
        ["Hall", "Kitchen"],
        {"Hall": 10.0, "Kitchen": 0.0},
    )
    assert w["Hall"] > w["Kitchen"]
    assert pytest.approx(sum(w.values()), rel=1e-6) == 1.0


def test_collect_transition_increments_winner_is_earliest_start() -> None:
    """After leaving A, first adjacent occupancy start in gap wins."""
    t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    gap = timedelta(seconds=60)
    intervals_by_area = {
        "A": [(t0, t0 + timedelta(minutes=5))],
        "Hall": [(t0 + timedelta(minutes=6), t0 + timedelta(minutes=20))],
        "Kitchen": [(t0 + timedelta(minutes=7), t0 + timedelta(minutes=20))],
    }
    inc = collect_transition_increments(
        "A",
        ["Hall", "Kitchen"],
        intervals_by_area,
        gap,
    )
    assert inc.get("Hall") == 1.0
    assert inc.get("Kitchen", 0) == 0.0


def test_collect_transition_increments_respects_gap() -> None:
    """Occupancy in B after gap window does not count."""
    t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    gap = timedelta(seconds=30)
    a_end = t0 + timedelta(minutes=5)
    intervals_by_area = {
        "A": [(t0, a_end)],
        "Hall": [(a_end + timedelta(minutes=2), t0 + timedelta(hours=1))],
    }
    inc = collect_transition_increments("A", ["Hall"], intervals_by_area, gap)
    assert inc == {}
