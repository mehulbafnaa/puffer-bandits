from __future__ import annotations

"""Aggregator shim for contextual/adversarial/neural agents.

This module re-exports the modular implementations under MAB_GPU.agents_ctx
to preserve backward compatibility for existing imports.
"""

from MAB_GPU.agents_ctx import (
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

