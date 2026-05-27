#!/usr/bin/env bash
set -euo pipefail
rm -f results/*_metrics.csv
ALGOS=(greedy dr_greedy lop go nrs random)
for a in "${ALGOS[@]}"; do
    PYTHONPATH=. python -m simulation.run_simulation --config configs/default.yaml --algorithm "$a" --seed 1 --results-tag "seed1"
    PYTHONPATH=. python -m simulation.run_simulation --config configs/default.yaml --algorithm "$a" --seed 1 --results-tag "seed2"
    PYTHONPATH=. python -m simulation.run_simulation --config configs/default.yaml --algorithm "$a" --seed 3 --results-tag "seed3"
    PYTHONPATH=. python -m simulation.run_simulation --config configs/default.yaml --algorithm "$a" --seed 4 --results-tag "seed4"
    PYTHONPATH=. python -m simulation.run_simulation --config configs/default.yaml --algorithm "$a" --seed 5 --results-tag "seed5"
done
python plots_script/comparison_all_metrics.py
