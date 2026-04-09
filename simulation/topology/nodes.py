import numpy as np

class Node:
    def __init__(self, id, node_type, position):
        self.id = id
        self.position = np.array(position)
        self.type = node_type  # 'sat', 'uav', 'ground', 'client'


def generate_nodes(num_sats=3, num_uavs=5, num_ground=10, area_size=2000):
    nodes = []
    # Satellites at fixed altitudes (LEO)
    for i in range(num_sats):
        x, y = np.random.uniform(0, area_size, 2)
        z = np.random.uniform(500, 2000)  # km
        nodes.append(Node(f"s{i}", "sat", (x, y, z)))
    
    # UAVs/HAPs
    for i in range(num_uavs):
        x, y = np.random.uniform(0, area_size, 2)
        z = np.random.uniform(20, 50)
        nodes.append(Node(f"u{i}", "uav", (x, y, z)))
    
    # Ground stations
    for i in range(num_ground):
        x, y = np.random.uniform(0, area_size, 2)
        z = 0
        nodes.append(Node(f"g{i}", "ground", (x, y, z)))
    
    return nodes
