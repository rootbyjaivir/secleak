## Audit card: cicids2017_tl

- rows: 2,830,743  features: 79
- real timestamps: **True** (cicids_dmy)
- exact-duplicate rows removable: **10.9%**; near-duplicate (4 s.f.): 13.2%
- label-conflicting rows (same features, different label): 0.25%
- test rows already in train, random split: **13.0%** (near: 15.6%)
- test rows already in train, temporal split: 8.5%

| top single-feature shortcuts | F1 (random) | F1 (temporal) |
|---|---|---|
| Packet Length Std | 0.783 | 0.747 |
| Packet Length Variance | 0.783 | 0.747 |
| Total Length of Bwd Packets | 0.738 | 0.732 |
| Subflow Bwd Bytes | 0.738 | 0.732 |
| Packet Length Mean | 0.728 | 0.617 |
