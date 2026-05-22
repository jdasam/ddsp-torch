import numpy as np


def get_scheduler(start_lr: float, stop_lr: float, decay_over: int):
    """Cosine LR decay lambda for torch.optim.lr_scheduler.LambdaLR.

    Returns a function step → multiplier (relative to start_lr).
    """
    def schedule(step: int) -> float:
        t = min(step / max(decay_over, 1), 1.0)
        cosine = (1 + np.cos(np.pi * t)) / 2
        return (stop_lr + (start_lr - stop_lr) * cosine) / start_lr
    return schedule
