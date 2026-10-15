# Rolling-origin evaluation (`secleak rolling`)

Purpose: give the temporal results an uncertainty estimate that reflects *different test windows*, and measure
the leakage directly with a matched control.

* Every raw label is cut into B (default 10) equal-count, time-ordered blocks.
* For each origin k from first_origin (default 4) to B-2 (the last block has no future, so its matched control
  would equal the past-only set and the delta would be 0 by construction): `rolling_past` trains on blocks < k and tests on block k.
* `blocked_matched` trains on a random sample from all OTHER blocks (past and future) with exactly the same
  number of rows per label as `rolling_past`, and is scored on the same block k.
* `delta = F1(blocked_matched) - F1(rolling_past)` is the gain from seeing the future at equal training size.
* Reported: per-origin F1 and delta, bootstrap 95% CI resampling origins (few origins => wide CI), min/max.

Limitations to state in the paper: models are untuned (sklearn defaults) to afford many folds; blocks are
per label so the label mix of each test block mirrors the dataset; rare labels may be absent from some blocks;
timestamps (and the CICIDS2017 AM/PM heuristic) determine the ordering; one dataset per run.

Post-hoc status: added after the first CICIDS2017 results (see the Deviations section of
docs/PREREGISTRATION.md); report it as exploratory next to the pre-registered analyses.
