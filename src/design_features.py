"""Build model features from configured sizing variables and transistor groups."""
from __future__ import annotations

from typing import Mapping

from src.selector import variable_names


def design_feature_values(sizes: Mapping[str, float], config: Mapping) -> dict[str, float]:
    """Return raw sizing variables and configured per-group geometry features."""
    values = {name: float(sizes[name]) for name in variable_names(config)}
    parameters = config.get("parameters", {})
    derived = config.get("optimization", {}).get(
        "derived_features", ("w_over_l", "w_times_l")
    )
    unknown_features = set(derived) - {"w_over_l", "w_times_l"}
    if unknown_features:
        raise ValueError(f"unsupported derived feature(s): {', '.join(sorted(unknown_features))}")
    for group_name, group in config.get("groups", {}).items():
        width_name, length_name = group["width"], group["length"]
        width_definition = parameters[width_name]
        length_definition = parameters[length_name]
        if width_name not in sizes and "default" not in width_definition:
            raise ValueError(f"group {group_name!r} width {width_name!r} has no sizing value")
        if length_name not in sizes and "default" not in length_definition:
            raise ValueError(f"group {group_name!r} length {length_name!r} has no sizing value")
        width = float(sizes.get(width_name, width_definition.get("default")))
        length = float(sizes.get(length_name, length_definition.get("default")))
        if width <= 0 or length <= 0:
            raise ValueError(f"group {group_name!r} has nonpositive W/L")
        if "w_over_l" in derived:
            values[f"{group_name}__w_over_l"] = width / length
        if "w_times_l" in derived:
            values[f"{group_name}__w_times_l"] = width * length
    return values


def design_feature_names(config: Mapping) -> tuple[str, ...]:
    """Return feature names in the same order used by design_feature_values."""
    parameters = config.get("parameters", {})
    names = list(variable_names(config))
    derived = config.get("optimization", {}).get(
        "derived_features", ("w_over_l", "w_times_l")
    )
    unknown_features = set(derived) - {"w_over_l", "w_times_l"}
    if unknown_features:
        raise ValueError(f"unsupported derived feature(s): {', '.join(sorted(unknown_features))}")
    for group_name, group in config.get("groups", {}).items():
        width_name, length_name = group["width"], group["length"]
        if width_name not in parameters or length_name not in parameters:
            raise ValueError(f"group {group_name!r} references an undefined parameter")
        if "w_over_l" in derived:
            names.append(f"{group_name}__w_over_l")
        if "w_times_l" in derived:
            names.append(f"{group_name}__w_times_l")
    return tuple(names)
