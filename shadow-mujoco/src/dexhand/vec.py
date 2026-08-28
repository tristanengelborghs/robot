"""Vectorized environments on threads, exploiting MuJoCo's GIL release.

The obvious ways to parallelize Python envs are subprocesses (robust, but pays
serialization on every step and multiplies memory by worker count) and a bare
loop (no parallelism). There is a third option specific to MuJoCo: mj_step
releases the GIL for the entire physics computation, and physics is ~95% of an
env step here (frame_skip 25 at 2 ms). So a plain ThreadPoolExecutor gets
near-linear speedup with zero IPC, shared memory by construction, and full
determinism — each env owns its model and data, so threads share nothing that
is written.

This file is also the designated Isaac Lab seam. Everything above it consumes
"step a batch of envs, get stacked dicts back"; a GPU-parallel backend replaces
this file and nothing else.

Autoreset follows the gymnasium convention: when an episode ends, the env is
reset immediately and the returned obs is the FIRST obs of the new episode;
the final obs of the finished one is in info["final_obs"]. Feeding the reset
obs to GAE as if the episode had continued corrupts the bootstrap — the buffer
must cut on done, which ppo.py does.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


class ThreadedVecEnv:
    def __init__(self, env_fns: Sequence[Callable], n_threads: Optional[int] = None):
        self.envs = [fn() for fn in env_fns]
        self.n = len(self.envs)
        self._pool = ThreadPoolExecutor(max_workers=n_threads or min(self.n, 8))
        e = self.envs[0]
        self.obs_keys = list(e.observation_space.spaces.keys())
        self.action_space = e.action_space
        self.observation_space = e.observation_space

    def reset(self, seeds: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
        seeds = seeds if seeds is not None else [None] * self.n
        results = list(self._pool.map(lambda p: p[0].reset(seed=p[1]), zip(self.envs, seeds)))
        return self._stack([r[0] for r in results])

    def step(self, actions: np.ndarray):
        def _one(args):
            env, act = args
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            if done:
                info = dict(info)
                info["final_obs"] = obs
                # bootstrap through time-limit truncation, cut at real failure
                info["terminated"] = term
                obs, _ = env.reset()
            return obs, r, done, info

        results = list(self._pool.map(_one, zip(self.envs, actions)))
        obs = self._stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([r[2] for r in results], dtype=np.float32)
        infos = [r[3] for r in results]
        return obs, rewards, dones, infos

    def _stack(self, obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        return {k: np.stack([o[k] for o in obs_list]) for k in self.obs_keys}

    def close(self):
        self._pool.shutdown(wait=False)
        for e in self.envs:
            e.close()
