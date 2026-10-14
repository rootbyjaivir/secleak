# Diagnostics: cicids2017_tl

## 1. Split composition and trivial baseline
`always_attack_f1` is the F1 of predicting 'attack' for every row. A model near this value on a split has learned nothing useful there.

| protocol | n_train | n_test | train_attack_frac | test_attack_frac | always_attack_f1 |
|---|---|---|---|---|---|
| random | 200,000 | 150,000 | 0.195 | 0.196 | 0.328 |
| temporal_global | 200,000 | 150,000 | 0.135 | 0.338 | 0.505 |
| temporal_per_class | 200,000 | 150,000 | 0.196 | 0.196 | 0.328 |
| temporal_per_label | 200,000 | 150,000 | 0.196 | 0.196 | 0.328 |

## 2. Fraction of each label classified correctly
For attack labels this is recall; for BENIGN it is specificity (1 - false-alarm rate). Missing cell = label absent from that test set.

| kind | label | random | temporal_global | temporal_per_class | temporal_per_label |
|---|---|---|---|---|---|
| attack | Bot | 0.488 | 0.007 | - | 0.495 |
| attack | DDoS | 0.925 | 0.642 | 0.629 | 0.909 |
| attack | DoS GoldenEye | 0.887 | - | - | 0.878 |
| attack | DoS Hulk | 0.868 | - | - | 0.871 |
| attack | DoS Slowhttptest | 0.886 | - | - | 0.799 |
| attack | DoS slowloris | 0.934 | - | - | 0.853 |
| attack | FTP-Patator | 0.759 | - | - | 0.765 |
| attack | Heartbleed | - | - | - | 0.500 |
| attack | Infiltration | 0.250 | 0.000 | - | 0.250 |
| attack | PortScan | 0.945 | 0.002 | 0.982 | 0.980 |
| attack | SSH-Patator | 0.503 | - | - | 0.503 |
| attack | Web Attack – Brute Force | 0.500 | - | - | 0.500 |
| attack | Web Attack – Sql Injection | 0.000 | - | - | 0.500 |
| attack | Web Attack – XSS | 0.500 | - | - | 0.500 |
| benign | BENIGN | 0.983 | 0.997 | 0.897 | 0.901 |

## 3. Gap sweep (temporal_per_class, gap between end of train and start of test)

| gap_frac | model | f1_attack | precision | recall |
|---|---|---|---|---|
| 0.000 | gbdt | 0.678 | 0.649 | 0.710 |
| 0.000 | logreg | 0.655 | 0.606 | 0.713 |
| 0.010 | gbdt | 0.652 | 0.637 | 0.667 |
| 0.010 | logreg | 0.654 | 0.604 | 0.712 |
| 0.050 | gbdt | 0.690 | 0.665 | 0.716 |
| 0.050 | logreg | 0.654 | 0.606 | 0.711 |
| 0.100 | gbdt | 0.680 | 0.648 | 0.714 |
| 0.100 | logreg | 0.660 | 0.614 | 0.715 |

> Temporal splits are deterministic, so seeds only vary the model. These tables describe one fixed partition of one dataset, not sampling uncertainty over data.
