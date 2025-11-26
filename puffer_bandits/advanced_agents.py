"""Aggregator shim for contextual/adversarial/neural agents.

This module re-exports the modular implementations under puffer_bandits.agents_ctx
to provide a single import surface for advanced agents.
"""

from .agents_ctx import (
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

