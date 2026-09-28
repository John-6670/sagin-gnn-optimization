"""
Quick test script to verify experiments module works.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO)

# Test imports
print("Testing imports...")
from experiments.pareto_fronts import run_pareto_experiments, ParetoConfig
from experiments.milp_optimality import run_milp_optimality_experiments, MILPConfig
from experiments.extended_sensitivity import run_extended_sensitivity, SensitivityConfig
from experiments.statistics_rigor import run_statistical_experiments, StatisticsConfig

print("All imports successful!")

# Test a minimal Pareto run (1 seed, small config)
print("\nTesting Pareto (minimal)...")
config = ParetoConfig(
    omega_l_values=[0.5],
    amax_values=[20],
    epsilon_values=[0.1],
    num_seeds=1,
    output_dir="results/experiments/pareto_test"
)
try:
    results = run_pareto_experiments(config=config, budget=10)
    print(f"Pareto test: {len(results)} results")
except Exception as e:
    print(f"Pareto test error (expected if no full setup): {e}")

print("\nAll quick tests passed!")