# SAGIN Placement & AirComp Simulation

This repository provides an academic prototype for **Space-Air-Ground Integrated Networks (SAGIN)** based on the paper:

- Function-aware server placement  
- Over-the-air aggregation (AirComp) modeling  
- SNR-driven utility optimization
- Two Timescale Optimization

The framework is designed to be modular and extensible, enabling gradual evolution toward more realistic channel models and learning-based optimization (e.g., GNN-guided placement).


## 📁 Repository Organization
- `configs/`  
  YAML configuration files controlling simulation, topology, and algorithm parameters.

- `simulation/`  
  Core environment:
  - `topology/` —  Node abstraction and hierarchical AirComp modeling (Satellite, UAV, Ground, Client)
  - `run_simulation.py` — main entry point for experiments

- `optimization/` — Bilevel optimization logic:
  - `placement.py` — Outer Loop (Slow timescale) server selection.
  - `objective.py` — AMSE-aware utility and compound objective computation.
  - `baselines.py` — Some baseline algorithms to compare with.

- `graph/`  
  Graph construction and feature extraction (prepared for GNN integration)

- `models/`  
  Prototype neural network models for future learning-based placement


## 🚀 Getting Started

### 1. Setup environment
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```
### 2. Install dependencies
Required libraries
```bash
pip install -r requirements.txt
```

### 3. Run a simulation
```bash
python -m simulation.run_simulation --config configs/default.yaml --budget 10
```


## ⚙️ Configuration Guide
All simulation parameters are defined in YAML files under `configs/`.

### Example structure:
```yaml
simulation:
  num_sats: 3
  num_uavs: 5
  num_ground: 10
  num_clients: 20
  area_size: 2000
  gradient_dim: 100
  sigma2: 10
  algorithm: test

algorithm:
  alpha: 0.5
  beta: 0.5
  budget: 10
  delta_list: [0.1, 0.2]
  snr_threshold: 0.0
```

### `simulation` Section
Controls the generated SAGIN topology:
- `num_sats` — number of satellites (LEO layer)
- `num_uavs` — number of UAV/HAP nodes
- `num_ground` — number of ground base stations
- `num_clients` — number of user devices
- `area_size` — size of simulation area (square)
- `gradient_dim` — gradient dimension of nodes
- `sigma2` — noise variance
- `algorithm` — select algorithm (greedy, go, lop, nrs, random)

### `algorithm` Section
Controls placement and optimization:
- `beta` — weight of AMSE in objective
- `budget` — total cost budget for server placement
- `delta_list` — per-tier error amplification factors
- `snr_threshold` — minimum SNR improvement required to accept a server


## 🧠 Core Concepts

### Two-Timescale Optimization

The framework utilizes a bilevel structure to handle network dynamics:
1. **Outer Loop (Slow Timescale):** Adapts server placement to long-term orbital trajectories and load trends. (will be added in future)
2. **Inner Loop (Fast Timescale):** Optimizes Over-the-Air control parameters (Power, Phase, Sync) to track instantaneous channel variations.

### Node abstraction & Orbital Mechanics

All entities are represented as a unified `Node` class. Satellites use **TLE (Two-Line Element)** data and the **SGP4** model for high-precision orbital propagation.
- Satellite
- UAV / HAP
- Ground station
- Client

### Over-the-Air Computation (AirComp)
Instead of digital aggregation, we exploit wireless superposition.
- **AMSE-Aware Placement:** Servers are placed where channel geometry minimizes the AirComp Mean Squared Error.
- **Hierarchical Aggregation:** Gradients flow from Clients → UAVs → Satellites → Gateways.

### Placement Strategies
- **Greedy (SNR-Aware):** Iteratively selects nodes based on marginal utility and SNR improvement.  
- **LOP (Latency-Only):** Traditional facility location focusing solely on delay.  
- **GO (Ground-Only):** Restricts placement to terrestrial infrastructure.  
- **NRS (Non-Robust Static):** Deterministic placement using expected values without SNR thresholds.
- **Random:** Randomly choose between available candiataes until either no more candidates exits or budget finishes.

### Objective Function
Placement is guided by a compound objective:
```
Cost = α · Latency + β · AMSE
```
where:
- Latency depends on distance
- AMSE depends on SNR and aggregation distortion


## 📊 Output & Metrics
The simulation evaluates:
- Total nodes and topology breakdown
- Selected servers
- Their types and positions


## 🔮 Roadmap
This framework is designed for incremental research development:
* ☑ Time-varying SNR with Doppler effects.
* ☑ Hierarchical AirComp modeling.
* ☐ GNN-guided placement acceleration.
* ☐ Wasserstein Distributionally Robust Optimization (DRO).