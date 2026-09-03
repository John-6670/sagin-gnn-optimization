# SAGIN Placement & AirComp Simulation

This repository is an academic prototype for **Space-Air-Ground Integrated Networks (SAGIN)** with a focus on:
- function-aware server placement,
- over-the-air aggregation (AirComp),
- SNR-driven utility optimization,
- robust placement via DRO,
- GNN-based candidate scoring.

The codebase supports simulation, baseline comparisons, GNN dataset generation/training, and robust placement experiments.

## 📁 Project Structure

- `configs/`
  - YAML experiment definitions for simulation topology, node types, algorithm parameters, and GNN settings.
- `simulation/`
  - `run_simulation.py` — main simulation runner and experiment orchestrator.
  - `config_loader.py` — YAML config loader.
  - `topology/` — node generation, AirComp modeling, patching logic.
  - `traffic/` — spatiotemporal traffic generation.
  - `environment/` — weather and channel environment.
  - `evaluation/` — performance metrics and result aggregation.
- `optimization/`
  - `placement.py` — core server selection logic, including GREEDY and DRO-GREEDY.
  - `objective.py` — AMSE and placement cost computation.
  - `baselines.py` — standard baselines: `lop`, `go`, `nrs`, `random`, `da`, and `dr_greedy`.
  - `dro.py` — distributionally robust optimization utilities.
  - `meta_learner.py` — meta-learning support for OTA control.
- `models/gnn/`
  - `precompute_dataset.py` — generate graph datasets for GNN training.
  - `train.py` — train the SAGIN hetero-GNN model.
  - `dataset.py` — dataset creation and dataset split helpers.
  - `hgnn.py` — heterogenous graph neural network model definitions.
- `precomputed_dataset/`
  - saved graph samples used for GNN training and inference.
- `checkpoints/`
  - saved GNN model checkpoints.

## 🚀 Quick Start

### 1. Install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run a simulation
```bash
python -m simulation.run_simulation --config configs/default.yaml --algorithm da --budget 20
```

### 3. Run all algorithms for comparison
```bash
python -m simulation.run_simulation --config configs/default.yaml --algorithm all
```

## 🧩 Configuration Type

The primary config file is `configs/default.yaml`.
It uses the following top-level sections:

- `simulation` — topology, timing, scenario count, and simulation engine settings.
- `nodes` — per-node-type parameters for satellites, UAVs, ground stations, and clients.
- `algorithm` — placement objective, budget, thresholds, and DRO/GNN hyperparameters.
- `gnn` — checkpoint path and inference settings used by the GNN model.

### Example `configs/default.yaml`
```yaml
simulation:
  num_sats: 36
  num_uavs: 8
  num_ground: 20
  num_clients: 500
  area_size: 2000
  gradient_dim: 100
  sigma2: 10
  algorithm: test
  num_scenarios: 8
  duration_hours: 24
  time_step_seconds: 60
  outer_interval_minutes: 10

nodes:
  satellite:
    altitude_km: 550.0
    inclination_deg: 53.0
    num_planes: 3
    sats_per_plane: 6
    power_w: 500.0
    coverage_radius_km: 1100.0
    motion: keplerian
  uav:
    altitude_km: 20.0
    power_w: 10.0
    station_keeping_km: 2.0
    deployment: hexagonal
    coverage_radius_km: 100.0
    motion: station_keeping
  ground:
    urban_macro:
      count: 6
      isd_m: 200
      height_m: 25
      antenna_gain_dbi: 17
      power_w: 40.0
      motion: stationary
    urban_micro:
      count: 3
      isd_m: 50
      height_m: 10
      antenna_gain_dbi: 5
      power_w: 10.0
      motion: stationary
    rural:
      count: 1
      isd_m: 17bi5ixbfbbbbbbbbbbberererererqop[eritq[poeruywoietuyqwerrrrrrrrrrqwoieiruiqwuu]]
      height_m: 35
      antenna_gain_dbi: 17
      power_w: 40.0
      motion: stationary
  client:
    stationary_fraction: 0.6
    pedestrian_fraction: 0.2
    vehicular_fraction: 0.2
    urban_fraction: 0.7
    rural_fraction: 0.3
    power_w: 0.1  \q]we[rt;y;iuop[y]\tryitoprtyoptr[yutuyyytitiroerr[rrrrrrrrrrrrrrrre[rptiouyw]peiortuy]rptiouy[repotiyuprtoy9uirtoiurotiuotiyyt pioutyrpotiryutryoitpuiorytuipoyrtpuiroyrytriyotryioptrypuiroyeyuioptuyiporyiepuotryeuioptryieouryepiutuyirpeutirpey treyupetyuipotyreuyieopertyuiprtieyp otruiptriyep uit iprtrt uipt ruioyrt eyrt eyiptr euipyt uiypr rueiopyt uprieoy eturiopyuip uit ueriopt eryt eryipu peorti upeoryy uipore yipuorpio yerpet ioruytir puyou iypptuiry oir oyuueitopry tr oyt oytiu opeytyieru oituyoptyytuyyyyyyyyyuyuyyuuyuyuyyu]]
algorithm:
  alpha: 0.5
  beta: 0.5
  budget: 20
  delta_list:
    - 0.1
    - 0.05
  snr_threshold: 0.0
  target_snr: 1.0

gnn:
  checkpoint: checkpoints/sagin_hgnn.pt
  kappa: 0.3
```

## 🎯 Available Simulation Algorithms

The following placement algorithms are available in `simulation/run_simulation.py`:
- `lop` — latency-only placement.
- `go` — ground-only greedy placement.
- `nrs` — non-robust greedy placement without SNR threshold.
- `random` — random server selection under budget.
- `da` — SNR+AMSE-aware deterministic adaptive selection.
- `dr_greedy` — distributionally robust greedy selection using sampled SNR scenarios.
- `all` / `test` — run every algorithm in the dictionary.

## 🔄 DRO Flow

The DRO flow in this repository works as follows:
1. sample multiple SNR scenarios to represent channel uncertainty.
2. compute robust marginal gains for candidate servers across scenarios.
3. greedily select servers while respecting the budget and SNR thresholds.
4. output a robust set of selected servers that trades off latency and AMSE.

This is implemented via `optimization.placement.dr_greedy_server_selection()` and exposed through `optimization.baselines.dr_selection()`.

## 🧪 Simulation Workflow

The end-to-end experiment flow is:

1. **Generate a precomputed GNN dataset**
   ```bash
   python models/gnn/precompute_dataset.py
   ```
   This creates `precomputed_dataset/sample_train_*.pt`, `sample_val_*.pt`, and `sample_test_*.pt` files.

2. **Train the GNN model**
   ```bash
   python models/gnn/train.py
   ```
   This loads precomputed dataset files, trains `SAGINHeteroGNN`, saves a checkpoint to `checkpoints/sagin_hgnn.pt`, and updates `configs/default.yaml`.

3. **Run the simulation**
   ```bash
   python -m simulation.run_simulation --config configs/default.yaml --algorithm dr_greedy --budget 20
   ```
   Use `--algorithm` to choose one of the supported strategies.

## 🛠️ Simulation CLI Options

`simulation/run_simulation.py` supports the following command-line arguments:
- `--config` — config file path (default: `configs/default.yaml`).
- `--budget` — override the placement budget.
- `--seed` — random seed (default: `123`).
- `--algorithm` — algorithm name or `all`.
- `--fl` — run federated learning experiments.
- `--task` — select a federated task when `--fl` is used.
- `--results-tag` — optional suffix for results filenames.
- `--sensitivity` — run DRO sensitivity analysis and ablation.

## ⚙️ Notes

- `configs/default.yaml` drives both simulation and placement.
- `models/gnn/train.py` writes the GNN checkpoint path back into `configs/default.yaml` under `gnn.checkpoint`.
- The GNN pipeline is separate from the core simulation; the simulation uses the GNN only if you enable or integrate it explicitly.

## ✅ Recommended Command Example

```bash
python -m simulation.run_simulation --config configs/default.yaml --algorithm da --budget 20 --seed 42
```

```bash
python -m simulation.run_simulation --config configs/default.yaml --algorithm dr_greedy --budget 20 --results-tag exp1
```
