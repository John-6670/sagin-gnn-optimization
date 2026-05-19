#!/usr/bin/env bash
set -euo pipefail
ALGOS=(greedy dr_greedy lop go nrs random)
for a in "${ALGOS[@]}"; do
    PYTHONPATH=. python -m simulation.run_simulation --config configs/default.yaml --algorithm "$a" --seed 1
    PYTHONPATH=. python -m simulation.run_simulation --config configs/default.yaml --algorithm "$a" --seed 2
    PYTHONPATH=. python -m simulation.run_simulation --config configs/default.yaml --algorithm "$a" --seed 3
    PYTHONPATH=. python -m simulation.run_simulation --config configs/default.yaml --algorithm "$a" --seed 4
    PYTHONPATH=. python -m simulation.run_simulation --config configs/default.yaml --algorithm "$a" --seed 5
done
python plots_script/comparison_all_metrics.py
