#!/usr/bin/env bash
# Full pipeline: venv -> deps -> GNN dataset -> GNN train -> simulation -> comparison
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

# --- 4. GNN dataset generation ---
echo "[gnn] generating precomputed dataset"
python models/gnn/precompute_dataset.py

# --- 5. GNN training ---
echo "[gnn] training GNN"
python models/gnn/train.py

# --- 6. Simulation ---
echo "[sim] running simulation"
python -m simulation.run_simulation --config configs/default.yaml --algorithm test --seed 42 --results-tag "1" --parallel

# --- 7. Comparison plots ---
echo "[compare] generating comparison plots"
python plots_script/comparison_all_metrics.py

echo "[done] full pipeline finished"
