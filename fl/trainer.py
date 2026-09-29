from __future__ import annotations
import copy
import csv
import logging
import os
import time
import numpy as np
import torch
from simulation.topology.patching import hybrid_patch, aircomp_aggregate

from fl.convergence import ConvergenceTracker, convergence_monitor
from .device import DEVICE

log = logging.getLogger("fl.trainer")


def _eval(model, test_loader, criterion):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            out = model(xb)
            l = criterion(out, yb)
            loss += l.item() * len(yb)
            total += len(yb)

            # Only compute accuracy for classification (targets are long integers)
            if yb.dtype == torch.long and out.ndim == 2:
                correct += (out.argmax(1) == yb).sum().item()

    acc = correct / max(total, 1)
    return loss / max(total, 1), acc


CONVERGENCE_TRACKER = ConvergenceTracker()


def _write_fl_csv(csv_path, round_data, header_written):
    """Append FL round metrics to CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not header_written:
            writer.writerow(['round', 'accuracy', 'loss', 'amse', 'wallclock_time', 'active_clients', 'p_t'])
        writer.writerow(round_data)

def FederatedRound(
    round_idx, clients, servers, ota_params, task, global_model,
    client_loaders, test_loader, snr_map, delta_list, use_hybrid=True,
    t_now=None, csv_path=None, header_written=False,
    client_sampling_rate: float = 1.0, rng: np.random.Generator = None
):
    """Run one FL round with wall-clock timing and optional CSV logging.

    Args:
        client_sampling_rate: Fraction of clients to sample per round (0.0-1.0).
            If < 1.0, randomly samples clients each round (FedAvg-style).
        rng: Random number generator for reproducible client sampling.
    """
    round_start = time.perf_counter()

    p_t = 1.0
    num_clients = len(clients)

    # Client sampling (FedAvg-style partial participation)
    if rng is None:
        rng = np.random.default_rng(round_idx)  # Deterministic per round

    if client_sampling_rate < 1.0:
        num_active = max(1, int(num_clients * client_sampling_rate))
        active = rng.choice(num_clients, num_active, replace=False).tolist()
        active.sort()
    else:
        active = list(range(num_clients))

    log.debug("  Round %d: %d/%d clients active (sampling_rate=%.2f)",
              round_idx, len(active), num_clients, client_sampling_rate)

    grads = []
    for cid in active:
        local = copy.deepcopy(global_model).to(DEVICE)
        if task.optimizer == 'sgd':
            opt = torch.optim.SGD(local.parameters(), lr=task.lr, **task.optimizer_kwargs)
        elif task.optimizer == 'adam':
            opt = torch.optim.Adam(local.parameters(), lr=task.lr, **task.optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer {task.optimizer}")

        local.train()
        epoch_losses = []
        for epoch in range(task.local_epochs):
            batch_loss = 0.0
            n_batches = 0
            for xb, yb in client_loaders[cid]:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                out = local(xb)
                loss = task.criterion(out, yb)
                loss.backward()
                if task.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(local.parameters(), task.grad_clip)
                opt.step()
                batch_loss += loss.item()
                n_batches += 1
            epoch_losses.append(batch_loss / max(n_batches, 1))
        log.debug("    client %d: epoch losses %s", cid, [f"{l:.4f}" for l in epoch_losses])

        diff = []
        for pg, pl in zip(global_model.parameters(), local.parameters()):
            diff.append((pg.data - pl.data).flatten())
        grad_vec = torch.cat(diff).detach().cpu().numpy()
        log.debug("    client %d: grad norm=%.6f", cid, float(np.linalg.norm(grad_vec)))
        grads.append(grad_vec)

    log.debug("  Round %d: aggregating %d gradients (use_hybrid=%s)...", round_idx, len(grads), use_hybrid)

    # Extract phase corrections from ota_params if available
    phase_corrections = {}
    if ota_params and isinstance(ota_params, dict):
        # ota_params structure: {server_id: {client_id: {"phase": ...}}}
        for server_params in ota_params.values():
            if isinstance(server_params, dict):
                for client_id, params in server_params.items():
                    if isinstance(params, dict) and "phase" in params:
                        phase_corrections[client_id] = params["phase"]

    if use_hybrid:
        _, amse_round, meta = hybrid_patch(
            [clients[i] for i in active], servers[0], snr_map, delta_list,
            phase_corrections=phase_corrections, t_now=t_now
        )
        w = np.array([meta['w_a'], meta['w_d']])
        base_gradient = np.mean(grads, axis=0)

        noise_std = np.sqrt(max(amse_round, 1e-12))
        aggregation_noise = np.random.normal(
            0.0,
            noise_std,
            size=base_gradient.shape,
        )

        g = (base_gradient + aggregation_noise) * w.sum()
        log.debug("  Round %d: hybrid_patch amse=%.6f  w_a=%.4f  w_d=%.4f  w_sum=%.4f",
                  round_idx, amse_round, meta['w_a'], meta['w_d'], w.sum())
    else:
        g, amse_round, _, _ = aircomp_aggregate(
            [clients[i] for i in active], servers[0], snr_map, delta_list,
            phase_corrections=phase_corrections, t_now=t_now
        )
        log.debug("  Round %d: aircomp amse=%.6f", round_idx, amse_round)

    log.debug("  Round %d: applying global model update (grad norm=%.6f)...",
              round_idx, float(np.linalg.norm(g)))
    ptr = 0
    for p in global_model.parameters():
        n = p.numel()
        upd = torch.tensor(g[ptr:ptr+n], dtype=p.dtype, device=DEVICE).view_as(p)
        p.data -= task.server_lr * upd
        ptr += n

    loss, acc = _eval(global_model, test_loader, task.criterion)
    log.debug("  Round %d: eval complete — loss=%.6f  acc=%.4f", round_idx, loss, acc)

    round_end = time.perf_counter()
    wallclock_time = round_end - round_start

    CONVERGENCE_TRACKER.update(
        round_idx,
        loss,
        acc,
        amse_round,
        wallclock_time,
    )

    # Optional CSV logging
    if csv_path is not None:
        _write_fl_csv(csv_path, [round_idx, acc, loss, amse_round, wallclock_time, len(active), p_t], header_written)

    return {
        'round': round_idx,
        'active_clients': len(active),
        'accuracy': acc,
        'loss': loss,
        'amse': amse_round,
        'p_t': p_t,
        'wallclock_time': wallclock_time,
    }


def run_fl_experiment(
    task_name: str,
    task,
    clients,
    servers,
    client_loaders,
    test_loader,
    delta_list,
    num_rounds: int = 20,
    use_hybrid: bool = True,
    t_now=None,
    csv_dir: str = "results/fl",
    algo_name: str = "dr_greedy",
    client_sampling_rate: float = 0.1,  # Default 10% client participation
    seed: int = 42,
):
    """
    Run full FL experiment for specified rounds with CSV logging.

    Args:
        task_name: Name of the FL task (e.g., 'cifar10', 'reddit_nwp', 'iot_anomaly')
        task: FLTask object
        clients: List of client nodes
        servers: List of selected server nodes (should be DR-Greedy selected)
        client_loaders: DataLoaders for each client
        test_loader: Test DataLoader
        delta_list: Quantization error deltas
        num_rounds: Number of FL rounds (default 20)
        use_hybrid: Whether to use hybrid patching
        t_now: Current simulation time
        csv_dir: Directory to save CSV results
        algo_name: Algorithm name for file naming
        client_sampling_rate: Fraction of clients to sample per round (0.0-1.0).
            Default 0.1 (10%) for realistic FL with many clients.
        seed: Random seed for reproducible client sampling.

    Returns:
        dict with final metrics and convergence logs
    """
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"fl_{task_name}_{algo_name}.csv")

    # Reset tracker for new experiment
    CONVERGENCE_TRACKER.rounds = []
    CONVERGENCE_TRACKER.losses = []
    CONVERGENCE_TRACKER.accuracies = []
    CONVERGENCE_TRACKER.amses = []
    CONVERGENCE_TRACKER.wallclock_times = []

    model = task.get_model()
    amse_hist, loss_hist, acc_hist, wallclock_hist = [], [], [], []

    log.info(f"[{task_name}/{algo_name}] Starting FL experiment: {num_rounds} rounds, {len(servers)} server(s), {len(clients)} clients, sampling_rate={client_sampling_rate:.2f}")

    # Compute SNR map ONCE (static topology - servers/clients don't move during FL)
    from skyfield.api import load
    ts = load.timescale()
    t_now_r = t_now if t_now is not None else ts.now()
    snr_map = {c: {servers[0]: c.compute_snr_to(servers[0], t_now_r)} for c in clients}

    rng = np.random.default_rng(seed)

    for r in range(num_rounds):
        header = (r == 0)
        res = FederatedRound(
            r, clients, servers, {}, task, model, client_loaders,
            test_loader, snr_map, delta_list, use_hybrid=use_hybrid,
            t_now=t_now_r, csv_path=csv_path, header_written=header,
            client_sampling_rate=client_sampling_rate, rng=rng
        )

        amse_hist.append(res['amse'])
        loss_hist.append(res['loss'])
        acc_hist.append(res['accuracy'])
        wallclock_hist.append(res['wallclock_time'])

        log.info("  [%s/%s] Round %2d/%d: acc=%.4f loss=%.6f amse=%.6f time=%.2fs (active=%d)",
                 task_name, algo_name, r+1, num_rounds, res['accuracy'], res['loss'], res['amse'], res['wallclock_time'], res['active_clients'])

    final_bound, logs = convergence_monitor(amse_hist, loss_hist, wallclock_hist)

    summary = CONVERGENCE_TRACKER.get_summary()
    summary.update({
        'task': task_name,
        'algorithm': algo_name,
        'final_bound': final_bound,
        'convergence_logs': logs,
    })

    log.info(f"[{task_name}/{algo_name}] FL Complete: Final Acc={acc_hist[-1]:.4f} Loss={loss_hist[-1]:.6f} Mean AMSE={np.mean(amse_hist):.6f} Total Time={summary['total_wallclock']:.1f}s")

    return summary
