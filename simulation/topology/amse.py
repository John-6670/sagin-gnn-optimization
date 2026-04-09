import numpy as np

def compute_amse(server, clients, snr_func):
    """
    Simplified AMSE for a server aggregating multiple clients.
    Uses Eq. (20) approximation.
    """
    d = 10  # gradient dimension (simplified)
    amse = 0.0
    for client in clients:
        snr = snr_func(server, client)
        amse += (1 / snr)  # inverse SNR contributes to error
    return amse * d