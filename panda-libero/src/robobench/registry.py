"""Model registry.

Adding your own architecture is two steps:

    # robobench/models/my_arch.py
    from robobench.registry import register_model
    from robobench.models.base import BasePolicy

    @register_model("my_arch")
    class MyArch(BasePolicy):
        ...

Then import it in `robobench/models/__init__.py` and set `model.name: my_arch`
in your config. Nothing else in the harness changes, which is the point: the
training loop, the evaluator and the statistics are identical across models, so a
difference in the results table is a difference in the architecture.
"""

from typing import Callable, Dict, Type

_MODEL_REGISTRY: Dict[str, Type] = {}


def register_model(name: str) -> Callable:
    def _wrap(cls):
        if name in _MODEL_REGISTRY:
            raise ValueError(
                f"model '{name}' is already registered by "
                f"{_MODEL_REGISTRY[name].__module__}.{_MODEL_REGISTRY[name].__name__}"
            )
        _MODEL_REGISTRY[name] = cls
        cls.registry_name = name
        return cls

    return _wrap


def build_model(name: str, **kwargs):
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"unknown model '{name}'. Registered: {sorted(_MODEL_REGISTRY)}.\n"
            "Did you import your module in robobench/models/__init__.py?"
        )
    return _MODEL_REGISTRY[name](**kwargs)


def list_models():
    return sorted(_MODEL_REGISTRY)
