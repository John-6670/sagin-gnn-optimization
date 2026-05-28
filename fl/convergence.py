import numpy as np


class ConvergenceTracker:
    def __init__(self):
        self.rounds = []
        self.losses = []
        self.accuracies = []
        self.amses = []

    def update(self, round_idx, loss, acc, amse):
        self.rounds.append(round_idx)
        self.losses.append(float(loss))
        self.accuracies.append(float(acc))
        self.amses.append(float(amse))

    def get_summary(self):
        return {
            'final_loss': self.losses[-1] if self.losses else None,
            'final_accuracy': self.accuracies[-1] if self.accuracies else None,
            'mean_amse': np.mean(self.amses) if self.amses else None,
            'rounds': len(self.rounds),
        }


def convergence_monitor(
    amse_history,
    loss_history,
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

        logs.append({
            "round": t,
            "amse": float(amse),
            "loss": float(loss),
            "theoretical_bound": float(theoretical_bound),
        })

    final_bound = (
        logs[-1]["theoretical_bound"]
        if logs
        else None
    )

    return final_bound, logs
