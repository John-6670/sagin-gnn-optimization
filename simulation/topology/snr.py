import numpy as np

def distance(node1, node2):
    return np.linalg.norm(node1.position - node2.position)

def compute_snr(node, client, tx_power=1.0, noise_power=1e-3, pathloss_exp=2.0):
    """
    Very simple pathloss-based SNR:
    SNR = P * (1 / distance^alpha) / noise
    """
    d = distance(node, client)
    snr = tx_power * (1 / (d**pathloss_exp)) / noise_power
    return snr