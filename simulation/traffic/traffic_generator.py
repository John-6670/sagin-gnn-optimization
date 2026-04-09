import numpy as np

def generate_traffic(num_clients, mean=1.0):
    return np.random.exponential(mean, size=num_clients)