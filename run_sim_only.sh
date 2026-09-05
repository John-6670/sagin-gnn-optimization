#!/usr/bin/env bash
# Simulation-only pipeline: venv -> deps -> simulation -> comparison
# Skips GNN dataset generation and training. Use when the GNN is already trained
# (checkpoint path is read from configs/default.yaml).
set -euo pipefail
cd "$(dirname "$0")"

# --- 1. Virtual environment ---
if [ -d ".venv" ]; then
    echo "[setup] .venv already exists, skipping creation"
else
    echo "[setup] creating .venv"
    python -m venv .venv
fi

# --- 2. Activate ---
source .venv/Scripts/activate

# --- 3. Dependencies ---
echo "[setup] installing requirements"
pip install -r requirements.txt

# --- 4. Simulation ---
echo "[sim] running simulation"
python -m simulation.run_simulation --config configs/default.yaml --algorithm test --seed 42 --results-tag "1" --parallel

# --- 5. Comparison plots ---
echo "[compare] generating comparison plots"
python plots_script/comparison_all_metrics.py

echo "[done] simulation pipeline finished"
