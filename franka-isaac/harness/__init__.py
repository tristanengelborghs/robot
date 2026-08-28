"""Driving the remote GPU box — the half of this project that runs on a laptop.

Isaac Sim needs an NVIDIA RTX GPU, so the simulator lives on a rented cloud
instance and the Isaac Lab extension itself is under ``source/insertion``, which
imports ``isaaclab`` and therefore only imports *there*. This package is
deliberately the opposite: pure Python with no simulator dependency, so it runs
and tests on the M1 with nothing installed and no instance running.

Everything here either builds a command to be executed on the GPU host, or does
arithmetic that has no business needing a GPU (retargeting, coordinate frames).
"""
