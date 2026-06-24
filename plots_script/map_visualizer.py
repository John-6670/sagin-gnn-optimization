import os
import matplotlib.pyplot as plt
from skyfield.api import load
import numpy as np
import sys

# Add the project root to the path so we can import from simulation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.config_loader import load_config
from simulation.topology.nodes import NodeType, generate_nodes

def visualize_map(config_path="configs/default.yaml", output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    config = load_config(config_path)
    num_sats = config["simulation"].get("num_sats", 1)
    num_uavs = config["simulation"].get("num_uavs", 2)
    num_ground = config["simulation"].get("num_ground", 4)
    num_clients = config["simulation"].get("num_clients", 20)
    area_size = config["simulation"].get("area_size", 2000)
    gradient_dim = config['simulation'].get('gradient_dim', 100)
    
    ts = load.timescale()
    t0 = ts.now()
    
    print("Generating nodes for visualization...")
    nodes = generate_nodes(
        num_sats=num_sats,
        num_uavs=num_uavs,
        num_ground=num_ground,
        num_clients=num_clients,
        area_size=area_size,
        gradient_dim=gradient_dim,
        t0=t0
    )
    
    # We will plot using the unprojected lat/lon. Since 1 deg ~ 111 km, we can plot lat/lon directly 
    # to show the spatial distribution.
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sat_lat, sat_lon = [], []
    uav_lat, uav_lon = [], []
    ground_lat, ground_lon = [], []
    client_lat, client_lon = [], []
    
    for n in nodes:
        lat, lon, _ = n.position
        if n.type == NodeType.SATELLITE:
            sat_lat.append(lat)
            sat_lon.append(lon)
        elif n.type == NodeType.UAV:
            uav_lat.append(lat)
            uav_lon.append(lon)
        elif n.type == NodeType.GROUND:
            ground_lat.append(lat)
            ground_lon.append(lon)
        elif n.type == NodeType.CLIENT:
            client_lat.append(lat)
            client_lon.append(lon)
            
    # Plot clients
    ax.scatter(client_lon, client_lat, c='blue', s=15, alpha=0.5, label='Clients', marker='o')
    # Plot ground
    ax.scatter(ground_lon, ground_lat, c='green', s=100, label='Ground BS', marker='s', edgecolors='black')
    # Plot UAVs
    ax.scatter(uav_lon, uav_lat, c='orange', s=150, label='UAVs', marker='^', edgecolors='black')
    
    # Filter satellites that are somewhat above the area
    half_deg = (area_size / 111.0) / 2.0
    visible_sat_lat = [lat for lat, lon in zip(sat_lat, sat_lon) if -half_deg*2 <= lat <= half_deg*2 and -half_deg*2 <= lon <= half_deg*2]
    visible_sat_lon = [lon for lat, lon in zip(sat_lat, sat_lon) if -half_deg*2 <= lat <= half_deg*2 and -half_deg*2 <= lon <= half_deg*2]
    
    if visible_sat_lat:
        ax.scatter(visible_sat_lon, visible_sat_lat, c='red', s=200, label='Satellites (Visible)', marker='*', edgecolors='black')

    # Draw Area Boundary
    rect = plt.Rectangle((-half_deg, -half_deg), half_deg*2, half_deg*2, fill=False, edgecolor='black', linestyle='--', linewidth=2, label='Simulation Area')
    ax.add_patch(rect)
    
    # Coverage circles for UAVs (Radius ~ 100km -> 100/111 deg)
    for lat, lon in zip(uav_lat, uav_lon):
        circle = plt.Circle((lon, lat), 100.0/111.0, fill=True, color='orange', alpha=0.1)
        ax.add_patch(circle)
        
    ax.set_title("SAGIN Network Topology Visualization")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Limit view to slightly larger than the area
    margin = half_deg * 0.2
    ax.set_xlim(-half_deg - margin, half_deg + margin)
    ax.set_ylim(-half_deg - margin, half_deg + margin)
    
    # Move legend outside
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "map_visualization.png")
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Map visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_map()
