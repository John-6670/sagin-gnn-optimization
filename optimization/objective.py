from simulation.topology.aircomp import compute_amse_kn


def weighted_compound_loss(latency, amse, alpha, beta):
    """Compute a weighted compound objective for placement evaluation."""
    return alpha * latency + beta * amse


def compute_utility(servers, clients, alpha, beta, delta_list):
    """
    servers: set/list of selected servers
    clients: list of client indices
    alpha, beta: weights
    """
    total_utility = 0.0

    for client in clients:
        best_cost = float("inf")

        for server in servers:
            latency = client.get_latency_to(server)
            amse = compute_amse_kn(client, server, delta_list)

            cost = alpha * latency + beta * amse

            if cost < best_cost:
                best_cost = cost

        if best_cost < float("inf"):
            total_utility += best_cost

    return total_utility
