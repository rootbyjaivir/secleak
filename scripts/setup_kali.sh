#!/usr/bin/env bash
# One-shot setup for Kali Linux (PEP 668 safe: uses a virtualenv).
set -euo pipefail
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git unzip
cd "$(dirname "$0")/.."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,parquet,xgb,llm]"
mkdir -p data/raw data/cache results
echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Smoke test:           secleak demo"
