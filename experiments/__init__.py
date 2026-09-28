"""
Experiments module for SAGIN GNN optimization paper.

Implements the 7 professor requirements:
1. OOD Robustness (optional)
2. Pareto Fronts with parameter sweeps
3. MILP Optimality Gap comparison
4. Extended Sensitivity analysis
5. Statistics Rigor (multi-seed, CI, runtime)
6. FL Metrics (wall-clock, 20 rounds, CSV) - DONE in fl/trainer.py
7. FL Integration at end of simulation
"""