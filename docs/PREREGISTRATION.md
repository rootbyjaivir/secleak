# SecLeak pre-registration (draft; edit, date and commit BEFORE running real data)

**Date frozen:** ____   **Git commit:** ____

## Research questions
RQ1 (prevalence). What fraction of published network-IDS papers using CIC-IDS2017, UNSW-NB15,
BoT-IoT and CICIoT2023 (a) split randomly, (b) deduplicate, (c) respect time, (d) test cross-dataset?
RQ2 (integrity). How much exact/near duplication and label conflict do these benchmarks contain, and what
share of test rows is already in train under a random split?
RQ3 (inflation). How much does attack-class F1 fall from the random-split protocol to
leak-controlled protocols (dedup, temporal, cross-dataset)?
RQ4 (shortcuts). Do single features reach near-perfect F1, and do they survive temporal shift?
RQ5 (memorisation). Do frontier LLMs reproduce dataset headers/rows/fields significantly above
a matched control?

## Hypotheses (directional, falsifiable)
H1. Mean F1 drop (random -> temporal_per_class), averaged over models, is >= 0.05 on >= 2 of 3 datasets
    with the bootstrap 95% CI excluding 0.
H2. >= 5% of rows are removable exact duplicates in at least one dataset.
H3. >= 1 dataset has a single feature with F1 >= 0.90 under random split.
H4. (exploratory, no directional claim) LLM memorisation: real-minus-control score > 0 with p < 0.01.
Failure to meet a hypothesis is reported, not hidden.

## Design (fixed in advance)
* Models: logreg, rf, gbdt, mlp (xgb optional). Same grid and tuning budget under every protocol;
  validation split is built with the SAME protocol as the test split.
* Preprocessing fitted on train only (sklearn Pipeline). Binary label (benign vs attack).
* Protocols: random, random_dedup, temporal_global, temporal_per_class (test_size 0.3), cross_dataset.
* Seeds: 10. Primary metric: attack-class F1. Secondary: macro-F1, MCC, precision, recall.
* Sensitivity analyses: test_size {0.2, 0.3, 0.4}; gap_frac {0, 0.02}; near-dup significant figures {3,4,5};
  dropping destination-port-like shortcut features.
* Exclusions: none after seeing results. Datasets without real timestamps are reported but flagged.

## Census protocol
Search queries, databases, date range and inclusion criteria are logged in `census/SEARCH_LOG.md`.
Two coders independently code >= 20% of papers with `census/codebook.md`; report Cohen's kappa.

## Known threats to validity
Time-proxy for CSVs lacking timestamps; AM/PM heuristic in CIC-IDS2017; label noise in the benchmarks
themselves; model-zoo choice; temporal cut-point arbitrariness; census sampling bias.


## Deviations and post-hoc additions (log every one; never edit the sections above)

**2026-10-14 - added protocol `temporal_per_label` (post-hoc).**
Reason: the first full CICIDS2017 run plus `secleak diagnose` showed that `temporal_per_class` splits within the
binary classes (benign / attack), so its test attacks are only the latest campaigns (DDoS and PortScan);
most attack types never appear in the test set. Its F1 drop therefore mixes drift within an attack type with
the arrival of attack types seen little or never in training. `temporal_per_label` cuts within each raw label
to separate the two. H1 and all other pre-registered analyses are still reported exactly as registered
(with `temporal_per_class`); `temporal_per_label` is reported as exploratory/post-hoc, alongside them.

**2026-10-14 - added `--cap-mode label_floor` and `--max-test-rows` (analysis options, default behaviour unchanged).**
Reason: with the default uniform caps (500k-row subsample, 200k training rows) rare labels have 1-2 test flows
(Infiltration, SQL Injection, Heartbleed) and almost no training flows, so their per-label recall is
uninformative. `label_floor` keeps every row of labels with <= 5000 rows in the training set (this shifts class
priors toward rare labels; it is applied identically to all protocols) and `--max-test-rows 0` scores all test
rows. Used ONLY for label-level diagnostics; headline protocol comparisons keep the uniform cap. Any result
produced with these options must say so.

**2026-10-15 - added `secleak rolling` (post-hoc).**
Reason: the 70/30 temporal splits are deterministic, so their CIs reflect model seeds only. Rolling-origin
evaluation (per-label time blocks, matched leaky control; see docs/ROLLING.md) gives uncertainty over test
windows and measures the benefit of seeing the future at equal training size. Reported as exploratory.
