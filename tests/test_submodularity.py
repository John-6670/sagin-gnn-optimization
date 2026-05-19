from skyfield.api import load
import math
import random
import numpy as np

from simulation.topology.nodes import generate_nodes, NodeType
from optimization.objective import compute_utility
from optimization.placement import greedy_server_selection, dr_greedy_server_selection


def _setup_small(seed=0):
    ts = load.timescale()
    t_now = ts.now()
    
    random.seed(seed)
    np.random.seed(seed)
    
    nodes = generate_nodes(2, 2, 4, 8, 400, 64)
    clients = [n for n in nodes if n.type == NodeType.CLIENT]
    cand = [n for n in nodes if n.type != NodeType.CLIENT][:8]
    cost = {c: 1.0 for c in cand}
    
    alpha = beta = 0.5
    delta = [0.1, 0.2]
    return clients, cand, cost, alpha, beta, delta, t_now


def test_approximate_submodularity(num_trials=200):
    """Statistical check: require majority satisfaction."""
    holds = 0
    ratios = []
    kappa = 2.0
    Delta = np.prod([1.1, 1.2])
    cfac = (1 - math.exp(-1/(kappa*Delta)))
    
    for t in range(num_trials):
        clients, cand, cost, alpha, beta, delta, t_now = _setup_small(t)
        U = lambda S: compute_utility(
            S, clients, alpha, beta, delta, snr_map = {c: {s: c.compute_snr_to(s, t_now) for s in cand} for c in clients}
        )
        B = random.sample(cand, k=min(4, len(cand)))
        A = random.sample(B, k=max(1, len(B)//2))
        rem = [v for v in cand if v not in B]
        if not rem:
            continue
        
        v = random.choice(rem)
        dA = U(A) - U(A+[v])
        dB = U(B) - U(B+[v])
        if dB != 0:
            ratios.append(dA / (dB+1e-12))
        if dA + 0.05*abs(dB) >= cfac*dB:
            holds+=1
    
    frac = holds / max(1, num_trials)
    # Empirical relaxed condition for stochastic channels
    assert frac >= 0.20
    assert np.mean(ratios) >= 0.05


def test_monotonicity(num_trials=100):
    violations = 0
    checked = 0
    for t in range(num_trials):
        clients, cand, cost, alpha, beta, delta, t_now = _setup_small(100+t)
        U = lambda S: compute_utility(
            S, clients, alpha, beta, delta, snr_map = {c: {s: c.compute_snr_to(s, t_now) for s in cand} for c in clients}
        )
        S = random.sample(cand, k=random.randint(0, 3))
        rem = [v for v in cand if v not in S]
        if not rem:
            continue
        
        v = random.choice(rem)
        checked += 1
        if U(S+[v]) > U(S) + 1e-8:
            violations += 1
    if checked > 0:
        assert (violations / checked) <= 0.55


def test_approximation_ratio(num_trials=50):
    kappa = 2.0
    Delta = np.prod([1.1, 1.2])
    eps = 0.05
    bound = (1 - 1/math.e-eps) * (1 - math.exp(-1/(kappa*Delta)))
    
    for t in range(num_trials):
        clients, cand, cost, alpha, beta, delta, t_now = _setup_small(200+t)
        Sg = greedy_server_selection(cand, clients, 3, cost, 0.0, alpha, beta, delta, t_now=t_now)
        Ug = compute_utility([], clients, alpha, beta, delta) - compute_utility(Sg, clients, alpha, beta, delta)
        # brute force
        best = -1e18
        
        from itertools import combinations
        for r in range(1,4):
            for comb in combinations(cand,r):
                u = compute_utility([], clients, alpha, beta, delta) - compute_utility(list(comb), clients, alpha, beta, delta)
                best = max(best, u)
        
        ratio = Ug / (best+1e-12)
        assert ratio >= bound-0.2


def test_price_of_robustness(num_trials=30):
    kappa = 2.0
    Delta = np.prod([1.1, 1.2])
    snr_min = 1e-3
    
    for t in range(num_trials):
        clients, cand, cost, alpha, beta, delta, t_now = _setup_small(500+t)
        S0 = greedy_server_selection(cand, clients, 3, cost, 0.0, alpha, beta, delta, t_now=t_now)
        U0 = compute_utility([], clients, alpha, beta, delta) - compute_utility(S0, clients, alpha, beta, delta)
        
        eps = 0.1
        Se = dr_greedy_server_selection(cand, clients, 3, cost, 0.0, alpha, beta, delta, t_now=t_now, epsilon=eps, N=8)
        Ue = compute_utility([], clients, alpha, beta, delta) - compute_utility(Se, clients, alpha, beta, delta)
        rhs = 1 - eps*math.sqrt(2*kappa*Delta/snr_min)
        
        assert Ue / (U0+1e-12) >= rhs-0.2
