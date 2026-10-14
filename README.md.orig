# SecLeak: integrity audit of network-intrusion-detection benchmarks

How much of the performance reported on CIC-IDS2017, UNSW-NB15, BoT-IoT (and friends) survives
*honest* evaluation? SecLeak measures four things and writes a paper-ready report:

1. **Duplication & label noise**: exact / near duplicates, label-conflicting rows, train/test overlap.
2. **Temporal leakage**: random vs. within-class-temporal vs. global-temporal vs. cross-dataset splits,
   same model zoo and same tuning budget, validation split built with the *same* protocol.
3. **Shortcuts**: single-feature classifiers, and whether they survive temporal shift.
4. **LLM memorisation**: header / row / feature-completion probes with matched controls and a
   permutation test (after Bordt et al., "Elephants Never Forget"). Optional; costs API credits.

Plus a **literature-census kit** (`census/`) and a **pre-registration** (`docs/PREREGISTRATION.md`).

## Install (Kali Linux)
```bash
git clone <your repo> secleak && cd secleak
bash scripts/setup_kali.sh        # apt deps + .venv + pip install -e .[dev,parquet,xgb,llm]
source .venv/bin/activate
make test                         # pytest
secleak demo                      # end-to-end on synthetic data with planted problems (~1 min)
```
Kali enforces PEP 668, so always use the venv (never `pip install` system-wide).

## Real-data workflow
```bash
# 0. put data in data/raw/ (see docs/DATASETS.md); record checksums
sha256sum data/raw/*/*/*.csv > data/CHECKSUMS.txt

# 1. audit cards (duplication, overlap, shortcuts) -> results/card_<dataset>.{json,md}
secleak audit cicids2017_tl --root . --max-rows 500000
secleak audit unsw_nb15_raw --root .
secleak audit bot_iot       --root .

# 2. the main experiment matrix (protocols x models x seeds) -> results/results_<dataset>.csv
secleak run cicids2017_tl --seeds 10
secleak run unsw_nb15_raw --seeds 10
secleak run bot_iot       --seeds 10

# 3. cross-dataset on harmonised features
secleak cross cicids2017_tl unsw_nb15_raw --seeds 5

# 4. merge + tables + figure -> results/report/{report.md,summary.csv,deltas.csv,f1_by_protocol.png}
secleak report results --out results/report

# 5. (optional) LLM memorisation probes; needs ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY=...        # SECLEAK_MODEL overrides the default model
secleak memorize data/raw/CICIDS2017/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv --n 40

# 6. census
secleak census census/census_template.csv --second census/coder2.csv
```
`--max-rows` subsamples uniformly *preserving time order*; duplication stats in the audit card are
best computed on the full data (`--max-rows 0` disables subsampling).

## Layout
```
src/secleak/  data.py protocols.py models.py experiments.py metrics.py report.py census.py cli.py synthetic.py
              audit/{dedup,shortcut,card,memorize}.py
configs/datasets.yaml     dataset presets + cross-dataset harmonisation
docs/                     DATASETS.md  PREREGISTRATION.md
census/                   codebook, template, search log
tests/                    pytest (splits, dedup, overlap, shortcuts, loaders, probes, census)
```

## Protocols
|----------------------|---------------------------------------------------|--------------------------------|-----------------------------------|
| name                 | train                                             | test                           | removes                           |
|----------------------|---------------------------------------------------|--------------------------------|-----------------------------------|
| `random`             | stratified random 70%                             | 30%                            | nothing (the literature default)  |
| `random_dedup`       | after dropping duplicates + label-conflict groups |                                | duplicate leakage                 |
| `temporal_global`    | earliest 70% of timeline                          | latest 30%                     | time leakage (changes class mix)  |
| `temporal_per_class` | earliest 70%  within each class                   | latest 30%                     | time leakage, keeps class balance |
| `cross_dataset`      | all of A                                          | all of B (harmonised features) | dataset-specific artefacts        |
|----------------------|---------------------------------------------------|--------------------------------|-----------------------------------|
## Caveats you must state in the paper
* Datasets without real timestamps (e.g. MachineLearningCVE CSVs) get a **row-order proxy**; flagged in reports.
* CIC-IDS2017 timestamp AM/PM fix is a heuristic; verify against pcaps.
* Deterministic temporal splits only vary model seeds; use many seeds and sensitivity analyses.
* LLM probes need a control, and a null result is a valid result.
* Report exactly which copy of each dataset you used (checksums).
* Frame findings as evaluation-practice issues, not accusations against individual papers.
