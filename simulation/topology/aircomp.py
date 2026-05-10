import numpy as np

from simulation.topology.nodes import Node


# AMSE(k, n)
def compute_amse_kn(client: Node, server: Node, delta_list, t_now=None):
    if t_now is not None:
        snr = client.compute_snr_to(server, t_now)
    else:
        snr = client.compute_snr_to(server)

    if snr <= 0.0:
        return float("inf")

    cascaded_error = np.prod([1 + d for d in delta_list])
    return (client.noise_variance * client.gradient_dim / snr) * cascaded_error


# AMSE(n)
def compute_amse_n(snr_dict, sigma2, d, sync_error=0.0):
    """
    snr_dict: dict {k: SNR(k,n)} for all clients k served by n
    sigma2: noise variance
    d: gradient dimension
    sync_error: synchronization error term
    """
    K = len(snr_dict)
    if K == 0:
        return 0.0

    inv_snr_sum = sum(1.0 / snr for snr in snr_dict.values())
    amse = (sigma2 * d / K) * inv_snr_sum + sync_error
    return amse
