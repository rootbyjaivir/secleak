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
For attack labels this is recall; for BENIGN it is specificity (1 - false-alarm rate). One table per model. `n_test` = test flows of that label (temporal_per_label / random split); labels with n<100 are too small to interpret. Missing cell = label absent from that test set.

### gbdt

| kind | label | n_test | random | temporal_global | temporal_per_class | temporal_per_label |
|---|---|---|---|---|---|---|
| attack | Bot | 605 | 0.939 | 0.001 | - | 0.166 |
| attack | DDoS | 38,408 | 0.999 | 0.632 | 0.627 | 0.999 |
| attack | DoS GoldenEye | 3,088 | 0.994 | - | - | 0.992 |
| attack | DoS Hulk | 69,669 | 1.000 | - | - | 0.995 |
| attack | DoS Slowhttptest | 1,683 | 0.999 | - | - | 0.947 |
| attack | DoS slowloris | 1,739 | 0.999 | - | - | 0.996 |
| attack | FTP-Patator | 2,386 | 0.999 | - | - | 0.997 |
| attack | Heartbleed (n<100) | 3 | 1.000 | - | - | 0.667 |
| attack | Infiltration (n<100) | 11 | 0.455 | 0.692 | - | 0.909 |
| attack | PortScan | 47,679 | 1.000 | 0.004 | 0.999 | 0.999 |
| attack | SSH-Patator | 1,769 | 1.000 | - | - | 0.988 |
| attack | Web Attack – Brute Force | 452 | 0.998 | - | - | 1.000 |
| attack | Web Attack – Sql Injection (n<100) | 10 | 0.900 | - | - | 0.833 |
| attack | Web Attack – XSS | 206 | 0.985 | - | - | 0.974 |
| benign | BENIGN | 681,929 | 0.999 | 1.000 | 0.931 | 0.908 |

### logreg

| kind | label | n_test | random | temporal_global | temporal_per_class | temporal_per_label |
|---|---|---|---|---|---|---|
| attack | Bot | 605 | 0.276 | 0.000 | - | 0.831 |
| attack | DDoS | 38,408 | 0.854 | 0.846 | 0.637 | 0.824 |
| attack | DoS GoldenEye | 3,088 | 0.786 | - | - | 0.797 |
| attack | DoS Hulk | 69,669 | 0.731 | - | - | 0.746 |
| attack | DoS Slowhttptest | 1,683 | 0.816 | - | - | 0.702 |
| attack | DoS slowloris | 1,739 | 0.814 | - | - | 0.724 |
| attack | FTP-Patator | 2,386 | 0.500 | - | - | 0.499 |
| attack | Heartbleed (n<100) | 3 | 0.667 | - | - | 1.000 |
| attack | Infiltration (n<100) | 11 | 0.364 | 0.000 | - | 0.091 |
| attack | PortScan | 47,679 | 0.890 | 0.001 | 0.975 | 0.961 |
| attack | SSH-Patator | 1,769 | 0.507 | - | - | 0.009 |
| attack | Web Attack – Brute Force | 452 | 0.000 | - | - | 0.000 |
| attack | Web Attack – Sql Injection (n<100) | 10 | 0.000 | - | - | 0.000 |
| attack | Web Attack – XSS | 206 | 0.000 | - | - | 0.000 |
| benign | BENIGN | 681,929 | 0.963 | 0.995 | 0.886 | 0.890 |

### rf

| kind | label | n_test | random | temporal_global | temporal_per_class | temporal_per_label |
|---|---|---|---|---|---|---|
| attack | Bot | 605 | 0.876 | 0.000 | - | 0.153 |
| attack | DDoS | 38,408 | 0.999 | 0.634 | 0.635 | 0.998 |
| attack | DoS GoldenEye | 3,088 | 0.995 | - | - | 0.971 |
| attack | DoS Hulk | 69,669 | 0.999 | - | - | 0.967 |
| attack | DoS Slowhttptest | 1,683 | 0.999 | - | - | 0.758 |
| attack | DoS slowloris | 1,739 | 0.998 | - | - | 0.954 |
| attack | FTP-Patator | 2,386 | 0.999 | - | - | 0.997 |
| attack | Heartbleed (n<100) | 3 | 1.000 | - | - | 1.000 |
| attack | Infiltration (n<100) | 11 | 0.545 | 0.769 | - | 0.818 |
| attack | PortScan | 47,679 | 0.994 | 0.001 | 0.998 | 0.998 |
| attack | SSH-Patator | 1,769 | 0.999 | - | - | 0.975 |
| attack | Web Attack – Brute Force | 452 | 0.995 | - | - | 0.998 |
| attack | Web Attack – Sql Injection (n<100) | 10 | 0.700 | - | - | 0.833 |
| attack | Web Attack – XSS | 206 | 0.956 | - | - | 0.944 |
| benign | BENIGN | 681,929 | 0.999 | 1.000 | 0.997 | 0.996 |

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
