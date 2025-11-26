"""Legacy shim: re-export from puffer_bandits.agents_ctx.

This module exists for backward compatibility only. Prefer:
    from puffer_bandits.agents_ctx import ...
"""

from puffer_bandits.agents_ctx import (  # type: ignore F401
    CtxAgentCfg,
    CtxAgent,
    pick_device,
    LinUCB,
    LinTS,
    EXP3,
    EXP3IX,
    NeuralTS,
    NeuralLinearTS,
)

__all__ = [
    "CtxAgentCfg",
    "CtxAgent",
    "pick_device",
    "LinUCB",
    "LinTS",
    "EXP3",
    "EXP3IX",
    "NeuralTS",
    "NeuralLinearTS",
]
