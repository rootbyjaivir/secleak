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
For attack labels this is recall; for BENIGN it is specificity (1 - false-alarm rate). One table per model. `n_test` = test flows of that label (temporal_per_label / random split); labels with n<100 are too small to interpret. Missing cell = label absent from that test set.

### gbdt

| kind | label | n_test | random | temporal_global | temporal_per_class | temporal_per_label |
|---|---|---|---|---|---|---|
| attack | Bot | 103 | 0.667 | 0.000 | - | 0.107 |
| attack | DDoS | 6,777 | 0.999 | 0.579 | 0.621 | 0.999 |
| attack | DoS GoldenEye | 548 | 1.000 | - | - | 0.972 |
| attack | DoS Hulk | 12,223 | 0.999 | - | - | 0.991 |
| attack | DoS Slowhttptest | 286 | 0.990 | - | - | 0.918 |
| attack | DoS slowloris | 310 | 1.000 | - | - | 0.993 |
| attack | FTP-Patator | 426 | 1.000 | - | - | 1.000 |
| attack | Heartbleed (n<100) | 1 | - | - | - | 0.000 |
| attack | Infiltration (n<100) | 2 | 0.000 | 0.000 | - | 0.500 |
| attack | PortScan | 8,413 | 1.000 | 0.004 | 1.000 | 0.999 |
| attack | SSH-Patator | 326 | 1.000 | - | - | 0.994 |
| attack | Web Attack – Brute Force (n<100) | 77 | 1.000 | - | - | 1.000 |
| attack | Web Attack – Sql Injection (n<100) | 1 | 0.000 | - | - | 1.000 |
| attack | Web Attack – XSS (n<100) | 36 | 1.000 | - | - | 1.000 |
| benign | BENIGN | 120,599 | 0.999 | 0.999 | 0.906 | 0.908 |

### logreg

| kind | label | n_test | random | temporal_global | temporal_per_class | temporal_per_label |
|---|---|---|---|---|---|---|
| attack | Bot | 103 | 0.310 | 0.015 | - | 0.883 |
| attack | DDoS | 6,777 | 0.850 | 0.704 | 0.636 | 0.819 |
| attack | DoS GoldenEye | 548 | 0.774 | - | - | 0.785 |
| attack | DoS Hulk | 12,223 | 0.737 | - | - | 0.751 |
| attack | DoS Slowhttptest | 286 | 0.783 | - | - | 0.680 |
| attack | DoS slowloris | 310 | 0.868 | - | - | 0.712 |
| attack | FTP-Patator | 426 | 0.519 | - | - | 0.531 |
| attack | Heartbleed (n<100) | 1 | - | - | - | 1.000 |
| attack | Infiltration (n<100) | 2 | 0.500 | 0.000 | - | 0.000 |
| attack | PortScan | 8,413 | 0.890 | 0.001 | 0.964 | 0.960 |
| attack | SSH-Patator | 326 | 0.006 | - | - | 0.013 |
| attack | Web Attack – Brute Force (n<100) | 77 | 0.000 | - | - | 0.000 |
| attack | Web Attack – Sql Injection (n<100) | 1 | 0.000 | - | - | 0.000 |
| attack | Web Attack – XSS (n<100) | 36 | 0.000 | - | - | 0.000 |
| benign | BENIGN | 120,599 | 0.966 | 0.994 | 0.887 | 0.894 |

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
