"""Legacy shim: re-export env classes from puffer_bandits.

Prefer importing from `puffer_bandits.envs` or the specific modules:
    from puffer_bandits.bandit_env import BernoulliBanditEnv
    from puffer_bandits.contextual_env import ContextualBanditEnv
"""

from puffer_bandits.bandit_env import BernoulliBanditEnv  # type: ignore F401
from puffer_bandits.contextual_env import ContextualBanditEnv  # type: ignore F401

__all__ = [
    "BernoulliBanditEnv",
    "ContextualBanditEnv",
]
