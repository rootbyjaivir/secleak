# Diagnostics: cicids2017_tl

## 1. Split composition and trivial baseline
`always_attack_f1` is the F1 of predicting 'attack' for every row. A model near this value on a split has learned nothing useful there.

| protocol | n_train | n_test | train_attack_frac | test_attack_frac | always_attack_f1 |
|---|---|---|---|---|---|
| random | 600,000 | 849,223 | 0.211 | 0.197 | 0.329 |
| temporal_global | 600,000 | 849,223 | 0.137 | 0.340 | 0.508 |
| temporal_per_class | 600,000 | 849,223 | 0.201 | 0.197 | 0.329 |
| temporal_per_label | 600,000 | 849,223 | 0.211 | 0.197 | 0.329 |

## 2. Fraction of each label classified correctly
For attack labels this is recall; for BENIGN it is specificity (1 - false-alarm rate). Missing cell = label absent from that test set.

| kind | label | random | temporal_global | temporal_per_class | temporal_per_label |
|---|---|---|---|---|---|
| attack | Bot | 0.697 | 0.000 | - | 0.383 |
| attack | DDoS | 0.951 | 0.704 | 0.633 | 0.940 |
| attack | DoS GoldenEye | 0.925 | - | - | 0.920 |
| attack | DoS Hulk | 0.910 | - | - | 0.903 |
| attack | DoS Slowhttptest | 0.938 | - | - | 0.802 |
| attack | DoS slowloris | 0.937 | - | - | 0.891 |
| attack | FTP-Patator | 0.833 | - | - | 0.831 |
| attack | Heartbleed | 0.889 | - | - | 0.889 |
| attack | Infiltration | 0.455 | 0.487 | - | 0.606 |
| attack | PortScan | 0.961 | 0.002 | 0.991 | 0.986 |
| attack | SSH-Patator | 0.835 | - | - | 0.657 |
| attack | Web Attack – Brute Force | 0.664 | - | - | 0.666 |
| attack | Web Attack – Sql Injection | 0.533 | - | - | 0.556 |
| attack | Web Attack – XSS | 0.647 | - | - | 0.639 |
| benign | BENIGN | 0.987 | 0.998 | 0.938 | 0.931 |

## 3. Gap sweep (temporal_per_class, gap between end of train and start of test)

| gap_frac | model | f1_attack | precision | recall |
|---|---|---|---|---|
| 0.000 | gbdt | 0.716 | 0.719 | 0.714 |
| 0.000 | logreg | 0.657 | 0.607 | 0.717 |
| 0.000 | rf | 0.831 | 0.983 | 0.720 |
| 0.010 | gbdt | 0.685 | 0.658 | 0.715 |
| 0.010 | logreg | 0.656 | 0.606 | 0.716 |
| 0.010 | rf | 0.831 | 0.982 | 0.720 |
| 0.050 | gbdt | 0.715 | 0.728 | 0.702 |
| 0.050 | logreg | 0.655 | 0.606 | 0.713 |
| 0.050 | rf | 0.831 | 0.983 | 0.720 |
| 0.100 | gbdt | 0.679 | 0.648 | 0.712 |
| 0.100 | logreg | 0.678 | 0.621 | 0.746 |
| 0.100 | rf | 0.831 | 0.983 | 0.720 |

> Temporal splits are deterministic, so seeds only vary the model. These tables describe one fixed partition of one dataset, not sampling uncertainty over data.
