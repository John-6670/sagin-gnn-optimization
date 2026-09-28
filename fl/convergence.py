import numpy as np
import time


class ConvergenceTracker:
    def __init__(self):
        self.rounds = []
        self.losses = []
        self.accuracies = []
        self.amses = []
        self.wallclock_times = []  # seconds per round

    def update(self, round_idx, loss, acc, amse, wallclock_time=None):
        self.rounds.append(round_idx)
        self.losses.append(float(loss))
        self.accuracies.append(float(acc))
        self.amses.append(float(amse))
        if wallclock_time is not None:
            self.wallclock_times.append(float(wallclock_time))

    def get_summary(self):
        return {
            'final_loss': self.losses[-1] if self.losses else None,
            'final_accuracy': self.accuracies[-1] if self.accuracies else None,
            'mean_amse': np.mean(self.amses) if self.amses else None,
            'rounds': len(self.rounds),
            'total_wallclock': float(np.sum(self.wallclock_times)) if self.wallclock_times else None,
            'mean_wallclock_per_round': float(np.mean(self.wallclock_times)) if self.wallclock_times else None,
        }


def convergence_monitor(
    amse_history,
    loss_history,
    wallclock_history=None,
    sigma2=1.0,
    rho=0.95,
    gamma=0.5,
):
    logs = []

    for t, (amse, loss) in enumerate(
        zip(amse_history, loss_history),
        start=1,
    ):

        theoretical_bound = (
            (rho ** t) * float(loss)
            + gamma * float(amse)
            + sigma2
        )

        log_entry = {
            "round": t,
            "amse": float(amse),
            "loss": float(loss),
            "theoretical_bound": float(theoretical_bound),
        }
        if wallclock_history is not None and t - 1 < len(wallclock_history):
            log_entry["wallclock_time"] = float(wallclock_history[t - 1])
        logs.append(log_entry)

    final_bound = (
        logs[-1]["theoretical_bound"]
        if logs
        else None
    )

    return final_bound, logs
