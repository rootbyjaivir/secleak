# Getting the datasets (Kali)

Put files under `data/raw/` exactly as below. Check each vendor page for the current download
location and licence; layouts change. Never commit raw data (`data/` is git-ignored).

| Preset | Expected path | Notes |
|---|---|---|
| `cicids2017_tl` | `data/raw/CICIDS2017/TrafficLabelling/*.csv` | "GeneratedLabelledFlows"; has `Timestamp`. Needs `cp1252` (set in config). Source: Canadian Institute for Cybersecurity (UNB), CIC-IDS2017 page. |
| `cicids2017_ml` | `data/raw/CICIDS2017/MachineLearningCVE/*.csv` | The copy most papers use. **No timestamps** -> only a row-order proxy; reports flag this. |
| `unsw_nb15_raw` | `data/raw/UNSW-NB15/UNSW-NB15_[1-4].csv` | Headerless raw files with `Stime`. The pre-split train/test CSVs have no time column; do not use them for temporal tests. |
| `bot_iot` | `data/raw/BoT-IoT/*.csv` | Full or 5% CSVs; has `stime`. Check column names with `head -1`. |

```bash
mkdir -p data/raw && cd data/raw && unzip ~/Downloads/<archive>.zip -d CICIDS2017
head -c 600 CICIDS2017/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv   # sanity check
```

## Known issues the loader handles (verify!)
* CIC-IDS2017 columns have leading spaces -> stripped; `Flow Bytes/s` has `inf`/`NaN` -> `NaN`.
* CIC-IDS2017 timestamps are `d/m/Y H:M` with **no AM/PM**; hours < 8 are shifted +12 as a
  *heuristic*. Confirm against pcap times before trusting within-day order; report it as a limitation.
* Different CSV copies of the same dataset differ (flow-construction fixes by later authors).
  Record the exact copy and SHA-256 in the paper: `sha256sum data/raw/**/*.csv > data/CHECKSUMS.txt`.

## Raw attack-type labels
`category_col` (UNSW-NB15 `attack_cat`, BoT-IoT `category`) feeds `Dataset.labels`, which the per-label
protocol (`temporal_per_label`) and `secleak diagnose` use. Without it they fall back to the binary label and
`temporal_per_label` becomes identical to `temporal_per_class`. UNSW-NB15 raw files contain label variants
(e.g. `Backdoor`/`Backdoors`, trailing spaces); the loader strips whitespace but does NOT merge variants:
inspect `label_counts.csv` and note any merge you apply.

## Adding a dataset
Add a preset in `configs/datasets.yaml` (`glob`, `label_col`, `benign_values` or `attack_value`,
`time_col` + `time_kind` or `file_order`, `drop_cols`) and, for cross-dataset tests, a `harmonise` map.
