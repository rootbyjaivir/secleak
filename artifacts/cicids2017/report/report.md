# SecLeak results

## Headline: mean F1 drop vs. random split

| dataset | protocol | mean_drop | min_drop | max_drop | rel_drop |
|---|---|---|---|---|---|
| cicids2017_tl | random_dedup | -0.0077 | -0.0303 | 0.0008 | -0.0091 |
| cicids2017_tl | temporal_global | 0.5471 | 0.3864 | 0.6764 | 0.5690 |
| cicids2017_tl | temporal_per_class | 0.2422 | 0.1661 | 0.3393 | 0.2527 |
| cicids2017_tl | temporal_per_label | 0.0972 | 0.0121 | 0.1597 | 0.1047 |

## Per-model summary (mean, 95% CI over seeds)

| dataset | protocol | model | mean | std | n | ci95 |
|---|---|---|---|---|---|---|
| cicids2017_tl | random | gbdt | 0.9972 | 0.0005 | 10 | 0.0004 |
| cicids2017_tl | random | logreg | 0.8450 | 0.0098 | 10 | 0.0070 |
| cicids2017_tl | random | mlp | 0.9790 | 0.0029 | 10 | 0.0021 |
| cicids2017_tl | random | rf | 0.9958 | 0.0003 | 10 | 0.0002 |
| cicids2017_tl | random_dedup | gbdt | 0.9968 | 0.0008 | 10 | 0.0005 |
| cicids2017_tl | random_dedup | logreg | 0.8753 | 0.0032 | 10 | 0.0023 |
| cicids2017_tl | random_dedup | mlp | 0.9808 | 0.0098 | 10 | 0.0070 |
| cicids2017_tl | random_dedup | rf | 0.9950 | 0.0002 | 10 | 0.0001 |
| cicids2017_tl | temporal_global | gbdt | 0.3208 | 0.1535 | 10 | 0.1098 |
| cicids2017_tl | temporal_global | logreg | 0.4586 | 0.0422 | 10 | 0.0302 |
| cicids2017_tl | temporal_global | mlp | 0.4111 | 0.0295 | 10 | 0.0211 |
| cicids2017_tl | temporal_global | rf | 0.4380 | 0.0012 | 10 | 0.0009 |
| cicids2017_tl | temporal_per_class | gbdt | 0.6579 | 0.1033 | 10 | 0.0739 |
| cicids2017_tl | temporal_per_class | logreg | 0.6564 | 0.0014 | 10 | 0.0010 |
| cicids2017_tl | temporal_per_class | mlp | 0.7044 | 0.0096 | 10 | 0.0069 |
| cicids2017_tl | temporal_per_class | rf | 0.8297 | 0.0004 | 10 | 0.0003 |
| cicids2017_tl | temporal_per_label | gbdt | 0.8373 | 0.0004 | 5 | 0.0005 |
| cicids2017_tl | temporal_per_label | logreg | 0.7239 | 0.0009 | 5 | 0.0011 |
| cicids2017_tl | temporal_per_label | rf | 0.9836 | 0.0024 | 5 | 0.0030 |

## Paired deltas (baseline - protocol), bootstrap 95% CI over seeds

| dataset | model | protocol | n_seeds | drop_mean | ci_lo | ci_hi | rel_drop |
|---|---|---|---|---|---|---|---|
| cicids2017_tl | gbdt | random_dedup | 10 | 0.0003 | -0.0002 | 0.0010 | 0.0003 |
| cicids2017_tl | logreg | random_dedup | 10 | -0.0303 | -0.0375 | -0.0249 | -0.0359 |
| cicids2017_tl | mlp | random_dedup | 10 | -0.0018 | -0.0063 | 0.0051 | -0.0018 |
| cicids2017_tl | rf | random_dedup | 10 | 0.0008 | 0.0006 | 0.0011 | 0.0008 |
| cicids2017_tl | gbdt | temporal_global | 10 | 0.6764 | 0.5926 | 0.7714 | 0.6783 |
| cicids2017_tl | logreg | temporal_global | 10 | 0.3864 | 0.3607 | 0.4073 | 0.4572 |
| cicids2017_tl | mlp | temporal_global | 10 | 0.5679 | 0.5519 | 0.5877 | 0.5801 |
| cicids2017_tl | rf | temporal_global | 10 | 0.5578 | 0.5572 | 0.5585 | 0.5602 |
| cicids2017_tl | gbdt | temporal_per_class | 10 | 0.3393 | 0.2997 | 0.4068 | 0.3403 |
| cicids2017_tl | logreg | temporal_per_class | 10 | 0.1887 | 0.1827 | 0.1941 | 0.2233 |
| cicids2017_tl | mlp | temporal_per_class | 10 | 0.2746 | 0.2683 | 0.2808 | 0.2805 |
| cicids2017_tl | rf | temporal_per_class | 10 | 0.1661 | 0.1658 | 0.1665 | 0.1668 |
| cicids2017_tl | gbdt | temporal_per_label | 5 | 0.1597 | 0.1592 | 0.1601 | 0.1601 |
| cicids2017_tl | logreg | temporal_per_label | 5 | 0.1198 | 0.1094 | 0.1273 | 0.1418 |
| cicids2017_tl | rf | temporal_per_label | 5 | 0.0121 | 0.0105 | 0.0145 | 0.0122 |

> Seeds vary model initialisation and the random split; deterministic temporal splits only vary the model, so CIs there are narrower than the true sampling uncertainty. Use >=10 seeds for the paper.