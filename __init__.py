"""GPU-enabled MAB advanced agents and runners using PufferLib.

API:
- Non-contextual advanced: ``KLUCB``, ``DiscountedUCB``, ``SlidingWindowUCB``
- Contextual/adversarial: ``LinUCB``, ``LinTS``, ``EXP3``, ``EXP3IX``, ``NeuralTS``, ``NeuralLinearTS``
- Configs: ``AgentCfg``, ``CtxAgentCfg``
- Utilities: ``pick_device``
- Factory: ``get_agent(name, cfg, **kwargs)``
"""

from .agents_ctx import (
    EXP3,
    EXP3IX,
    CtxAgentCfg,
    LinTS,
    LinUCB,
    NeuralLinearTS,
    NeuralTS,
)
from .agents import (
    KLUCB,
    Agent,
    AgentCfg,
    DiscountedUCB,
    SlidingWindowUCB,
)
from .utils.device import pick_device

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AgentCfg",
    "CtxAgentCfg",
    "Agent",
    "KLUCB",
    "DiscountedUCB",
    "SlidingWindowUCB",
    "LinUCB",
    "LinTS",
    "EXP3",
    "EXP3IX",
    "NeuralTS",
    "NeuralLinearTS",
    "pick_device",
    "get_agent",
]


def get_agent(name: str, cfg: AgentCfg, /, **kwargs) -> Agent:
    """Factory to build an agent by short name.

    Examples:
        get_agent("klucb", cfg, kl_alpha=3.0)
        get_agent("ducb", cfg, c=2.0, discount=0.99)
        get_agent("swucb", cfg, c=2.0, window=200)
    """
    key = name.strip().lower()
    if key in {"klucb"}:
        return KLUCB(cfg, **kwargs)
    if key in {"ducb", "discounted_ucb"}:
        return DiscountedUCB(cfg, **kwargs)
    if key in {"swucb", "sliding_window_ucb"}:
        return SlidingWindowUCB(cfg, **kwargs)
    raise ValueError(f"Unknown agent name: {name}")
