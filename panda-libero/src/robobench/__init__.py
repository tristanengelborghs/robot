"""robobench — a minimal harness for benchmarking new visuomotor architectures on LIBERO.

Design goal: the only file you should need to touch to test a new architecture is a
single module under `robobench/models/`. Everything else — data, training loop,
rollout evaluation, statistics — is shared so that comparisons stay apples-to-apples.
"""

__version__ = "0.1.0"

from robobench.registry import register_model, build_model, list_models  # noqa: F401
