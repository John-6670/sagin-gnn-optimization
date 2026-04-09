# SAGIN GNN Optimization

This repository provides an academic prototype for Space-Air-Ground Integrated Networks (SAGIN) placement and over-the-air aggregation analysis.

## Repository organization

- `configs/` — YAML configuration for simulation parameters, network settings, and algorithms.
- `simulation/` — modular environment components, including topology, network models, traffic, and energy estimation.
- `optimization/` — placement and objective utilities for greedy and robust decision-making.
- `graph/` — graph builder and feature extraction stubs for future GNN development.
- `models/` — prototype neural network models for graph-based placement acceleration.

## Getting started

1. Create and activate your virtual environment.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run a small simulation:
   ```bash
   python -m simulation.run_simulation --config configs/default.yaml --budget 3
   ```

## Notes

- This codebase is designed for incremental refinement: start with simple topology and objective models, then extend to 3GPP-aligned channel modeling, AirComp-aware placement, and GNN-guided search.
- The current implementation uses a simplified path-loss-based SNR model and a prototype greedy placement heuristic.
