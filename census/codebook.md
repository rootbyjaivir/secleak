# Census codebook

One row per (paper, dataset) pair in `census/census_template.csv`.
Code from the paper text/appendix/code repo. If not stated, code `unclear` (never guess `yes`).

| column | values | definition |
|---|---|---|
| paper_id | string | stable key (first-author+year+letter) |
| year, venue | | publication year, venue/preprint server |
| dataset | cicids2017 / unsw-nb15 / bot-iot / ciciot2023 / other | benchmark used for the main result |
| split_method | random / stratified-random / kfold-random / temporal / predefined / cross-dataset / unclear | how test data was separated; `predefined` = vendor's train/test files |
| dedup | yes / no / unclear | duplicates removed *before* splitting, explicitly stated |
| temporal_aware | yes / no / unclear | training data strictly earlier than test data |
| cross_dataset | yes / no | additionally evaluated on a different dataset |
| llm_based | yes / no | LLM/foundation model used as detector, feature extractor or labeller |
| code_available | yes / no | public code or data-processing scripts |
| reported_f1 | float 0-1 | headline F1/accuracy-equivalent; put the metric name in `metric` |
| metric | f1 / acc / other | |
| notes | text | |

Rules: k-fold with shuffling = `kfold-random`. Accuracy-only papers: code `reported_f1` empty.
Disagreements are resolved by discussion; report kappa on the pre-resolution codes.
