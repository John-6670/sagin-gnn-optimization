# SAGIN Placement & AirComp Simulation

This repository provides an academic prototype for **Space-Air-Ground Integrated Networks (SAGIN)** with a focus on:

- Function-aware server placement  
- Over-the-air aggregation (AirComp) modeling  
- SNR-driven utility optimization  

The framework is designed to be modular and extensible, enabling gradual evolution toward more realistic channel models and learning-based optimization (e.g., GNN-guided placement).


## 📁 Repository Organization
- `configs/`  
  YAML configuration files controlling simulation, topology, and algorithm parameters.

- `simulation/`  
  Core environment:
  - `topology/` — Node abstraction (satellite, UAV, ground, client)
  - `run_simulation.py` — main entry point for experiments

- `optimization/`  
  Placement and objective logic:
  - Greedy server selection
  - AMSE and utility computation

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

algorithm:
  alpha: 0.5
  beta: 0.5
  budget: 20
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

### `algorithm` Section
Controls placement and optimization:
- `beta` — weight of AMSE in objective
- `budget` — total cost budget for server placement
- `delta_list` — per-tier error amplification factors
- `snr_threshold` — minimum SNR improvement required to accept a server


## 🧠 Core Concepts

### Node abstraction

All entities are represented as a unified `Node` class:
- Satellite
- UAV / HAP
- Ground station
- Client

Each node includes:
- 3D position
- Channel model (Rayleigh + pathloss)
- SNR computation
- Latency estimation

### Objective Function
Placement is guided by a compound objective:
```
Cost = α · Latency + β · AMSE
```
where:
- Latency depends on distance
- AMSE depends on SNR and aggregation distortion

### Greedy Placement
Servers are selected iteratively based on:
- Utility improvement
- Deployment cost
- SNR gain threshold

This approximates submodular optimization under budget constraints.


## 📊 Output
Running a simulation prints:
- Total nodes and topology breakdown
- Selected servers
- Their types and positions


## 📝 Notes
- Channel model currently uses:
   - Distance-based pathloss
   - Rayleigh fading
- SNR is precomputed per client-server pair
- Utility is evaluated using AMSE-aware formulation


## 🔮 Future Extensions
This framework is designed for incremental research development:
- Multi-hop AirComp (client → UAV → satellite)
- Time-varying SNR with Doppler effects
- Distributionally robust optimization (DRO)
- GNN-guided placement acceleration