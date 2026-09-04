# SAGIN Placement & AirComp Simulation

This repository is an academic prototype for **Space-Air-Ground Integrated Networks (SAGIN)** with a focus on:
- Function-aware server placement
- Over-the-air aggregation (AirComp)
- SNR-driven utility optimization
- Robust placement via Distributionally Robust Optimization (DRO)
- GNN-based candidate scoring

The codebase supports simulation, baseline comparisons, GNN dataset generation/training, and robust placement experiments.

## 📁 Project Structure

```
├── configs/
│   └── default.yaml          # Main experiment configuration
├── simulation/
│   ├── run_simulation.py     # Main simulation runner & experiment orchestrator
│   ├── config_loader.py      # YAML config loader
│   ├── topology/
│   │   ├── nodes.py          # Node classes, Walker constellation, client mobility
│   │   ├── aircomp.py        # Hierarchical AirComp AMSE computation
│   │   ├── constellation.py  # Walker delta constellation generation
│   │   └── patching.py       # Hybrid patching for quantization bounds
│   ├── traffic/
│   │   └── traffic_generator.py  # Spatiotemporal traffic generation
│   ├── environment/
│   │   └── weather.py        # Two-state weather Markov chain
│   ├── evaluation/
│   │   └── metrics.py        # E2E latency, energy, CVaR computation
│   └── network/
│       └── channel_model.py  # SAGIN channel model (Rician fading, pathloss)
├── optimization/
│   ├── placement.py          # Core server selection: greedy, DR-greedy, OTA control
│   ├── objective.py          # Composite objective (latency + AMSE), marginal gains
│   ├── baselines.py          # Baseline algorithms: LOP, GO, NRS, Random, DA, DR, FedSN, HSFL
│   ├── dro.py                # Wasserstein DRO with proper dual reformulation
│   └── meta_learner.py       # MAML inner optimizer for OTA power control
├── models/gnn/
│   ├── precompute_dataset.py # Generate graph datasets (Latin hypercube sampling)
│   ├── train.py              # Train SAGINHeteroGNN with submodularity + SNR penalties
│   ├── dataset.py            # PyG HeteroData dataset & splits
│   └── hgnn.py               # Heterogeneous GNN (HeteroConv + GATConv)
├── precomputed_dataset/      # Saved graph samples for GNN training
├── checkpoints/              # GNN model checkpoints (.pt + .meta.pkl)
├── plots/                    # Generated comparison plots
└── results/                  # CSV metrics per algorithm
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run a simulation (static test mode)
```bash
python -m simulation.run_simulation --config configs/default.yaml --algorithm dr_greedy --budget 50 --seed 42
```

### 3. Run all algorithms for comparison
```bash
python -m simulation.run_simulation --config configs/default.yaml --algorithm all --seed 42
```

### 4. Dynamic mode (time-evolving, duration > 0 in config)
```bash
python -m simulation.run_simulation --config configs/default.yaml --algorithm all --duration 1
```

## 🧩 Configuration

Primary config: `configs/default.yaml`

### Key sections:
| Section | Description |
|---------|-------------|
| `simulation` | Topology counts, area, timing, gradient_dim, sigma2, num_scenarios, duration_hours, time_step_seconds, outer_interval_minutes |
| `nodes` | Per-node-type parameters (satellite, UAV, ground, client) |
| `algorithm` | alpha, beta, budget, delta_list, snr_threshold, target_snr, epsilon_wasserstein, tau_amse, alpha_cvar, coherence_time, sigma_snr, num_scenarios |
| `gnn` | checkpoint path, kappa (pruning fraction), epochs, batch_size, lr, loss weights |
| `channel` | Carrier frequency, bandwidth, link Rician K-factors & coherence times |
| `weather` | Atmospheric loss for clear/rain states |
| `node_distribution` | Client & ground BS distribution fractions |

### Node costs (in `run_simulation.py:build_costs()`):
| Tier | Cost |
|------|------|
| Satellite | 10.0 |
| UAV/HAP | 5.0 |
| Ground BS | 2.0 |

Budget default: **50**

## 🎯 Available Simulation Algorithms

Run with `--algorithm <name>`:
| Algorithm | Description |
|-----------|-------------|
| `lop` | Latency-only placement (minimizes propagation latency) |
| `go` | Ground-only greedy (constrains to ground BSs) |
| `nrs` | Non-robust greedy without SNR threshold |
| `random` | Random server selection under budget |
| `da` | SNR+AMSE-aware deterministic adaptive selection |
| `fedsn` | FedSN-inspired: prioritizes satellites with compute/feasibility proxies |
| `hsfl` | Hierarchical Split FL baseline (UAV edge + Sat global) |
| `dr_greedy` | Distributionally robust greedy using sampled SNR scenarios (Wasserstein DRO) |
| `all` / `test` | Run every algorithm for comparison |

## 🔄 DRO Flow (Proper Wasserstein Dual)

The DRO implementation now uses a **correct Wasserstein dual reformulation** per Eq. 27:

1. **Sample N SNR scenarios** with OU temporal perturbation (orbital-average for satellites, instantaneous for HAP/ground)
2. **Compute per-scenario marginal gains** `gain_i = Cost(S) - Cost(S ∪ {v})` using hierarchical AMSE
3. **Estimate Lipschitz constants** `L_i` from scenario dispersion in log-SNR space
4. **Solve the dual** `min_λ≥0 { λε + (1/N) Σ_i [gain_i + L_i²/(4λ)] }` with closed-form optimum `λ* = sqrt(mean(L²) / (4ε))`
5. **Robust gain** = `dual(λ*)` blended with empirical CVaR: `min(robust_gain, CVaR_α(gains))`
6. **Cost-normalized greedy selection** (gain/cost) + local 1-swap refinement

This replaces the previous linear approximation that collapsed to boundary solutions.

## 🧪 GNN Pipeline

### 1. Generate precomputed dataset (~14k samples, Latin hypercube)
```bash
python models/gnn/precompute_dataset.py
```
Outputs: `precomputed_dataset/sample_train_*.pt`, `sample_val_*.pt`, `sample_test_*.pt`

### 2. Train GNN model
```bash
python models/gnn/train.py
```
- Architecture: `SAGINHeteroGNN` (HeteroConv + GATConv, hidden=128, 3 layers, 4 heads)
- Loss: MSE + submodularity regularizer + SNR gradient penalty
- Scheduler: CosineAnnealingLR, early stopping (patience=20)
- Outputs: `checkpoints/sagin_hgnn.pt` + `checkpoints/sagin_hgnn.meta.pkl`
- Updates `configs/default.yaml` `gnn.checkpoint` path

### 3. Run simulation with GNN-enhanced selection
```bash
python -m simulation.run_simulation --config configs/default.yaml --algorithm dr_greedy --budget 50
```
GNN scores candidates by marginal utility gain; downweights satellites (0.6×) and UAVs (0.85×).

## 🛠️ Simulation CLI Options

| Argument | Description |
|----------|-------------|
| `--config` | Config file path (default: `configs/default.yaml`) |
| `--budget` | Override placement budget |
| `--seed` | Random seed (default: `123`) |
| `--algorithm` | Algorithm name or `all` |
| `--fl` | Run federated learning experiments |
| `--task` | FL task name (`reddit_nwp`, `cifar10`, `iot_anomaly`) |
| `--results-tag` | Optional suffix for results filenames |
| `--sensitivity` | Run DRO sensitivity analysis & ablation |

## ⚙️ Simulation Modes

### Static Mode (`duration_hours = 0`)
1. Generate nodes → Build candidates/clients
2. Run algorithm → Select servers
3. Compute OTA metrics (AMSE_n, AMSE_kn)
4. Generate comparison plots in `plots/`

### Dynamic Mode (`duration_hours > 0`)
1. Generate nodes → Initialize weather
2. For each time step (60s default): generate traffic → update channels → OTA control → compute AMSE/energy/latency/CVaR
3. Re-select servers at outer interval (30 min default)
4. Save CSV metrics per algorithm in `results/`

### Key Algorithms
| Algorithm | Signature |
|-----------|-----------|
| All baselines | `(candidates, clients, budget, cost, thresh, alpha, beta, delta_list, N, t_now=None, **kwargs)` |
| Random | Accepts `seed` kwarg (re-randomizes each outer loop) |
| DR-greedy | Extra: `epsilon`, `alpha_cvar`, `coherence_time`, `sigma_snr`, `gnn_checkpoint`, `kappa`, `tau_amse` |

## 🔑 Key Implementation Details

1. **Node types**: `SATELLITE`, `UAV`, `GROUND`, `CLIENT` (Enum in `nodes.py`)
2. **Position handling**: Skyfield for satellite orbital propagation; HAPs use station-keeping box; clients have mobility models (60% stationary, 20% pedestrian 1 m/s, 20% vehicular 15 m/s)
3. **Channel model**: `SaginChannelModel` computes SNR with Rician fading per link type
4. **AMSE computation**: Hierarchical AirComp in `aircomp.py` — tier-specific cascaded errors
5. **Hybrid patching**: `patching.py` for quantization bounds in hierarchical aggregation
6. **GNN input features**: Position (3), velocity (3), SNR stats (4: mean/max/min/std), load (1), one-hot type (4) = 15 features
7. **DRO parameters**: `N` scenarios, `ε` (ambiguity radius), `α_CVaR` (0.95), coherence_time, `σ_snr`
6. **All algorithms re-select** at each outer interval (client mobility + orbital motion)

## 📊 Output Files

- `plots/` — AMSE comparisons, grouped bars, algorithm benchmarks
- `results/<algo>_metrics.csv` — Per-step metrics: latency, AMSE, energy, CVaR@95/90/99, tier counts
- `checkpoints/sagin_hgnn.pt` + `.meta.pkl` — GNN checkpoint & metadata (input dims, normalization)

## ✅ Recommended Commands

```bash
# Static comparison (all algorithms)
python -m simulation.run_simulation --config configs/default.yaml --algorithm all --seed 42

# DR-greedy with GNN pruning (kappa=0.6)
python -m simulation.run_simulation --config configs/default.yaml --algorithm dr_greedy --budget 50 --seed 42

# DRO sensitivity analysis
python -m simulation.run_simulation --config configs/default.yaml --algorithm dr_greedy --budget 50 --sensitivity

# Federated learning experiment
python -m simulation.run_simulation --config configs/default.yaml --fl --task reddit_nwp --algorithm da
```