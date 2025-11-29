
import torch


@torch.no_grad()
def exp3_probs(w: torch.Tensor, gamma: float) -> torch.Tensor:
    """EXP3 mixed probabilities from weights.

    w: (n, k)
    returns p: (n, k)
    """
    W = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    k = w.shape[1]
    return (1.0 - gamma) * (w / W) + gamma / float(k)


@torch.no_grad()
def exp3ix_probs(w: torch.Tensor) -> torch.Tensor:
    """EXP3-IX uses only normalized weights without explicit mixing at sampling."""
    W = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return w / W


@torch.no_grad()
def exp3_update_factor(p_sel: torch.Tensor, rewards: torch.Tensor, eta: float, k: int) -> torch.Tensor:
    """Compute multiplicative factor to update selected weights: exp(eta * r_hat / k)."""
    r_hat = rewards / p_sel.clamp_min(1e-12)
    return torch.exp(eta * r_hat / float(k))
