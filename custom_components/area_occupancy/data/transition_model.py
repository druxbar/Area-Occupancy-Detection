"""Learned room-to-room transition counts and prior-boost weighting."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from homeassistant.util import dt as dt_util

from ..const import DEFAULT_LOOKBACK_DAYS
from ..time_utils import from_db_utc, to_db_utc

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

# Laplace smoothing for P(to | from) over configured adjacents only.
_LAPLACE_ALPHA = 0.25


def weights_for_adjacent_targets(
    adjacent_names: list[str],
    raw_counts: dict[str, float],
    *,
    laplace_alpha: float = _LAPLACE_ALPHA,
) -> dict[str, float]:
    """Normalize boost weights over adjacents; uniform if no counts."""
    names = [n for n in adjacent_names if n]
    if not names:
        return {}
    counts = [max(0.0, float(raw_counts.get(n, 0.0))) for n in names]
    total = sum(counts)
    if total < 1e-9:
        u = 1.0 / len(names)
        return dict.fromkeys(names, u)
    denom = total + laplace_alpha * len(names)
    return {n: (c + laplace_alpha) / denom for n, c in zip(names, counts, strict=True)}


def _earliest_occupancy_start_in_gap(
    intervals_b: list[tuple[datetime, datetime]],
    after: datetime,
    before: datetime,
) -> datetime | None:
    best: datetime | None = None
    for b_s, _b_e in intervals_b:
        if b_s > after and b_s <= before:
            if best is None or b_s < best:
                best = b_s
    return best


def collect_transition_increments(
    from_area_name: str,
    adjacent_names: list[str],
    intervals_by_area: dict[str, list[tuple[datetime, datetime]]],
    max_gap: timedelta,
) -> dict[str, float]:
    """Count one transition per end of occupied interval in *from* to earliest adjacent start in gap."""
    increments: dict[str, float] = {}
    int_a = intervals_by_area.get(from_area_name, [])
    for _a_s, a_e in int_a:
        window_end = a_e + max_gap
        winner_name: str | None = None
        winner_t: datetime | None = None
        for b_name in adjacent_names:
            if b_name == from_area_name:
                continue
            b_int = intervals_by_area.get(b_name, [])
            t = _earliest_occupancy_start_in_gap(b_int, a_e, window_end)
            if t is not None and (winner_t is None or t < winner_t):
                winner_t = t
                winner_name = b_name
        if winner_name is not None:
            increments[winner_name] = increments.get(winner_name, 0.0) + 1.0
    return increments


def _load_intervals_map(
    coordinator: AreaOccupancyCoordinator,
    area_names: list[str],
    period_start: datetime,
    period_end: datetime,
) -> dict[str, list[tuple[datetime, datetime]]]:
    """Load merged occupied-interval cache rows filtered by config entry."""
    db = coordinator.db
    entry_id = coordinator.entry_id
    out: dict[str, list[tuple[datetime, datetime]]] = {}
    try:
        with db.get_session() as session:
            for name in area_names:
                ps = to_db_utc(period_start)
                pe = to_db_utc(period_end)
                rows = (
                    session.query(db.OccupiedIntervalsCache)
                    .filter_by(entry_id=entry_id, area_name=name)
                    .filter(db.OccupiedIntervalsCache.start_time < pe)
                    .filter(db.OccupiedIntervalsCache.end_time > ps)
                    .order_by(db.OccupiedIntervalsCache.start_time)
                    .all()
                )
                out[name] = [
                    (from_db_utc(r.start_time), from_db_utc(r.end_time)) for r in rows
                ]
    except (SQLAlchemyError, ValueError, TypeError, RuntimeError, OSError) as e:
        _LOGGER.error("Failed to load intervals for transition model: %s", e)
        return {n: [] for n in area_names}
    return out


def run_transition_learning(
    coordinator: AreaOccupancyCoordinator,
    now: datetime | None = None,
) -> None:
    """Decay stored counts, add observations from cached occupied intervals, upsert DB."""
    now = now or dt_util.utcnow()
    areas = list(coordinator.areas.values())
    if not any(a.config.transition_learn_enabled for a in areas):
        return

    decay = coordinator.integration_config.transition_learn_decay
    decay = max(0.0, min(1.0, float(decay)))
    db = coordinator.db
    entry_id = coordinator.entry_id
    lookback_days = DEFAULT_LOOKBACK_DAYS
    period_end = now
    period_start = now - timedelta(days=lookback_days)
    area_names = [a.area_name for a in areas]
    intervals_by_area = _load_intervals_map(coordinator, area_names, period_start, period_end)

    merged: dict[tuple[str, str], float] = {}
    for area in areas:
        if not area.config.transition_learn_enabled:
            continue
        adj = area.config.adjacent_area_names()
        if len(adj) < 1:
            continue
        gap_s = int(area.config.transition_learn_max_gap)
        gap = timedelta(seconds=max(1, gap_s))
        inc = collect_transition_increments(
            area.area_name, adj, intervals_by_area, gap
        )
        for to_name, v in inc.items():
            key = (area.area_name, to_name)
            merged[key] = merged.get(key, 0.0) + v

    try:
        with db.get_session() as session:
            if decay < 1.0:
                session.execute(
                    text(
                        "UPDATE area_transition_counts SET "
                        "transition_count = transition_count * :decay, "
                        "updated_at = :updated "
                        "WHERE entry_id = :eid"
                    ),
                    {
                        "decay": decay,
                        "updated": to_db_utc(now),
                        "eid": entry_id,
                    },
                )
            atc = db.AreaTransitionCounts
            for (from_n, to_n), add_v in merged.items():
                if add_v <= 0:
                    continue
                row = (
                    session.query(atc)
                    .filter_by(
                        entry_id=entry_id,
                        from_area_name=from_n,
                        to_area_name=to_n,
                    )
                    .first()
                )
                if row:
                    row.transition_count = float(row.transition_count) + add_v
                    row.updated_at = to_db_utc(now)
                else:
                    session.add(
                        atc(
                            entry_id=entry_id,
                            from_area_name=from_n,
                            to_area_name=to_n,
                            transition_count=float(add_v),
                            updated_at=to_db_utc(now),
                        )
                    )
            session.execute(
                text(
                    "DELETE FROM area_transition_counts WHERE entry_id = :eid "
                    "AND transition_count < :eps"
                ),
                {"eid": entry_id, "eps": 1e-6},
            )
            session.commit()
    except (SQLAlchemyError, ValueError, TypeError, RuntimeError, OSError) as e:
        _LOGGER.error("Transition learning DB update failed: %s", e)
