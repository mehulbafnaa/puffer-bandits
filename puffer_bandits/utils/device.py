
import os
from typing import Any

import torch

try:
    import psutil
except Exception:  # pragma: no cover - optional
    psutil = None  # type: ignore


def pick_device(preferred: str | None = None) -> torch.device:
    """Select a torch device according to availability and user hint.

    :param preferred: Explicit device string (e.g., ``"cuda"``, ``"mps"``, ``"cpu"``). If ``None``, auto‑detect.
    :type preferred: str or None
    :return: Torch device to use for computation.
    :rtype: torch.device
    """
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    """Synchronize device for accurate profiling timings."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def memory_stats(device: torch.device) -> dict[str, Any]:
    """Return lightweight memory stats depending on backend.

    - CUDA: allocated/reserved bytes
    - MPS: current allocated memory if available
    - CPU: process RSS via psutil

    :param device: Torch device to query.
    :type device: torch.device
    :return: Dictionary of memory metrics; keys depend on backend.
    :rtype: dict[str, Any]
    """
    if device.type == "cuda":
        try:
            return {
                "allocated": int(torch.cuda.memory_allocated(device)),
                "reserved": int(torch.cuda.memory_reserved(device)),
            }
        except Exception:
            return {}
    if device.type == "mps":
        try:
            # Available on recent PyTorch; may not exist on older builds
            alloc = int(torch.mps.current_allocated_memory())  # type: ignore[attr-defined]
            return {"allocated": alloc}
        except Exception:
            return {}
    # CPU fallback: process RSS
    if psutil is not None:
        try:
            rss = int(psutil.Process(os.getpid()).memory_info().rss)
            return {"rss": rss}
        except Exception:
            return {}
    return {}
