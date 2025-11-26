from __future__ import annotations

import numpy as np
import torch

try:
    from .agents import Agent
except Exception:  # type: ignore
    Agent = object  # type: ignore


def _desc(x) -> str:
    if isinstance(x, torch.Tensor):
        return f"torch[{x.dtype}] on {x.device} shape={tuple(x.shape)}"
    if isinstance(x, np.ndarray):
        return f"numpy[{x.dtype}] (CPU) shape={x.shape}"
    return f"{type(x).__name__}"


def print_agent_state_devices(agent: object) -> None:
    print("[device] agent device:", getattr(agent, "device", "<unknown>"))
    for name in ("Q", "N", "alpha", "beta"):
        if hasattr(agent, name):
            t = getattr(agent, name)
            if isinstance(t, torch.Tensor):
                print(f"[device] agent.{name}:", _desc(t))


class AgentDebugWrapper(Agent):  # type: ignore[misc]
    """Wraps an Agent and prints device info on first calls."""

    def __init__(self, inner: Agent):  # type: ignore[override]
        # Do not call super(); we forward attributes to inner
        self._inner = inner
        self._printed_select = False
        self._printed_update = False

    # Expose expected attributes
    def __getattr__(self, name):  # fallback to inner
        return getattr(self._inner, name)

    @torch.no_grad()
    def select_actions(self, t: int):  # type: ignore[override]
        a = self._inner.select_actions(t)
        if not self._printed_select:
            print("[device] select_actions -> actions:", _desc(a))
            self._printed_select = True
        return a

    @torch.no_grad()
    def update(self, actions, rewards, *args, **kwargs):  # type: ignore[override]
        out = self._inner.update(actions, rewards, *args, **kwargs)
        if not self._printed_update:
            print("[device] update <- actions:", _desc(actions))
            print("[device] update <- rewards:", _desc(rewards))
            self._printed_update = True
        return out


def wrap_agent_for_debug(agent: Agent) -> Agent:  # type: ignore[override]
    return AgentDebugWrapper(agent)
