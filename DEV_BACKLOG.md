## Area Occupancy Detection — backlog

Small, reviewable chunks. Each item: goal, notes, likely touch points.

### Transition model (room-to-room smoothing)
- **Goal**: learn \(P(next_area | current_area)\) from motion sequences; apply transient boosts to adjacent areas to avoid brief “no presence” gaps.
- **Status (v1)**: `area_transition_counts` table + analysis step 5 learns from occupied-interval cache (opt-in per area); boosts split `transition_boost_logit` by Laplace-smoothed counts over configured adjacents; global EMA decay in integration options.
- **Notes**: keep opt-in; prefer simple Markov counts + time window + decay; avoid loops with “All Areas”.
- **Touch points**: `data/transition_model.py`, `coordinator.py` (`_apply_transition_boost`, `async_refresh_transition_counts`), `data/analysis.py`, `db/schema.py` / `db/queries.py`.
- **Follow-ups**: richer sequence model, cap boosts, diagnostics matrix export, tune gap vs boost window defaults.

### Context priors (weekday/weekend + sun)
- **Goal**: priors incorporate context: weekday vs weekend, sun above/below horizon, maybe season.
- **Notes**: start with weekday/weekend split (minimal data) then add sun state (binary).
- **Touch points**: `custom_components/area_occupancy/data/prior.py`, `custom_components/area_occupancy/time_utils.py`, config options for enable + weights.

### Explainability UI (timeline)
- **Goal**: “why occupied now / why dropped” timeline: top contributors over last N minutes, threshold crossings, decay starts/stops.
- **Notes**: store small ring buffer in memory per area; expose via diagnostic attributes; optional service to dump.
- **Touch points**: `custom_components/area_occupancy/area/area.py`, `custom_components/area_occupancy/sensor.py`, `custom_components/area_occupancy/service.py`.

### Recorder / data health diagnostics
- **Goal**: show per-sensor sample counts (last 24h, last 7d), last interval sync time, recorder availability.
- **Notes**: helps “no samples / correlation failed” troubleshooting.
- **Touch points**: `custom_components/area_occupancy/db/sync.py`, `custom_components/area_occupancy/db/queries.py`, `custom_components/area_occupancy/diagnostics.py`, `custom_components/area_occupancy/data/health.py`.

### Export / import learned state
- **Goal**: export/import priors, correlations, learned multipliers (JSON) for backup/migration.
- **Notes**: include versioned schema; avoid exporting raw intervals by default.
- **Touch points**: `custom_components/area_occupancy/service.py`, `custom_components/area_occupancy/db/*`.

### Auto-weight calibration UX
- **Goal**: preview changes (diff) + lock weights + rollback to last snapshot.
- **Notes**: keep safe: no silent changes unless enabled; store snapshots.
- **Touch points**: `custom_components/area_occupancy/data/analysis.py` auto weight step, `custom_components/area_occupancy/number.py` weight entities, new persistence for snapshots.

