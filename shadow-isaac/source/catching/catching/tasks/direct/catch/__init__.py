"""Registration for the catch task."""

import gymnasium as gym

from . import agents

gym.register(
    id="Catch-Shadow-Direct-v0",
    entry_point=f"{__name__}.catch_env:CatchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.catch_env_cfg:CatchEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
