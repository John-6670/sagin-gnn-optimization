#!/usr/bin/env bash
set -e

echo "Running a smoke simulation with default config..."
python -m simulation.run_simulation --config configs/default.yaml --budget 30
