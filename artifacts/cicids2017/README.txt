Commands (dataset cicids2017_tl, 2,830,743 rows after loader fix; checksums in CHECKSUMS.txt)
results_matrix_a.csv           secleak run cicids2017_tl --seeds 10
                               (defaults: --max-rows 500000 uniform subsample, --max-train-rows 200000, uniform cap,
                                protocols random/random_dedup/temporal_global/temporal_per_class, models logreg/rf/gbdt/mlp)
results_matrix_per_label.csv   secleak run cicids2017_tl --protocols temporal_per_label --models logreg rf gbdt --seeds 5
                               (post-hoc protocol, see docs/PREREGISTRATION.md deviations)
report/                        secleak report <dir with both CSVs above> (merged)
diagnostics_full/              secleak diagnose cicids2017_tl --max-rows 0 --max-train-rows 600000 --max-test-rows 0
                               --cap-mode label_floor --models rf gbdt logreg --seeds 1  (single seed, untuned, label_floor cap)
diagnostics_other_models/      secleak diagnose cicids2017_tl --models gbdt logreg --seeds 1  (500k subsample, uniform caps)
Caveats: temporal splits are deterministic (seeds vary only the model); rare labels (Heartbleed, Infiltration,
SQL Injection) have <=36 flows in the whole dataset and must not be interpreted.
