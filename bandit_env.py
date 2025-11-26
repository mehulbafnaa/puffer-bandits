"""Legacy shim: re-export the Gymnasium bandit env from puffer_bandits.

Prefer importing from:
    from puffer_bandits.bandit_env import BernoulliBanditEnv
"""

from puffer_bandits.bandit_env import *  # type: ignore F401,F403
