
import torch


@torch.no_grad()
def safe_cholesky(mat: torch.Tensor, jitter: float = 1e-6, max_tries: int = 5) -> torch.Tensor:
    """Batched safe Cholesky with jitter escalation and PSD repair fallback.

    Returns a lower-triangular factor with same batch dims as `mat`.
    """
    assert mat.size(-1) == mat.size(-2), "matrix must be square"
    eye = torch.eye(mat.size(-1), device=mat.device, dtype=mat.dtype)
    for i in range(max_tries):
        scale = (10.0 ** i) * jitter
        try:
            L, info = torch.linalg.cholesky_ex(mat + scale * eye)
            if torch.all(info == 0):
                return L
        except Exception:
            pass
    # Final fallback: eigen clamp to make PSD
    w, V = torch.linalg.eigh(mat)
    w_clamped = torch.clamp(w, min=jitter)
    M_psd = V @ torch.diag_embed(w_clamped) @ V.mH
    return torch.linalg.cholesky(M_psd)
