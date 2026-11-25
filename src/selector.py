"""Design-space helpers shared by the RL controller and sizing runner."""
from __future__ import annotations

from typing import Mapping


def variable_names(sizing_config: Mapping) -> tuple[str, ...]:
    """Return the validated ordering of RL-controlled sizing variables."""
    optimization = sizing_config.get("optimization", {})
    if not isinstance(optimization, Mapping):
        raise ValueError("sizing config 'optimization' must be a mapping")
    names = optimization.get("variables")
    if not isinstance(names, list) or not names:
        raise ValueError("sizing config must define a non-empty optimization.variables list")
    if not all(isinstance(name, str) for name in names):
        raise ValueError("optimization.variables entries must be strings")
    if len(set(names)) != len(names):
        raise ValueError("optimization.variables must not contain duplicates")
    unknown = set(names) - set(sizing_config.get("parameters", {}))
    if unknown:
        raise ValueError(
            "optimization variables are not defined as parameters: "
            + ", ".join(sorted(unknown))
        )
    return tuple(names)
