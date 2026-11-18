"""Generate analysis-specific SPICE decks from a netlist and its YAML recipe."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

from scripts.netlist_tools import resize_netlist


def render_transient_deck(
    template: str,
    sizes: Mapping[str, float],
    sizing_config: Mapping,
    waveform_name: str,
) -> str:
    deck = resize_netlist(template, sizes, sizing_config, literal_wl=True)
    pattern = re.compile(r"(?im)^(\s*wrdata\s+)\S+")
    deck, count = pattern.subn(lambda match: match.group(1) + waveform_name, deck)
    if count != 1:
        raise ValueError(f"expected one transient wrdata command, found {count}")
    return deck


def render_gain_deck(
    template: str,
    sizes: Mapping[str, float],
    sizing_config: Mapping,
    waveform_name: str,
) -> str:
    gain_config = sizing_config.get("analyses", {}).get("gain", {})
    if not gain_config.get("enabled", False):
        raise ValueError("the sizing config does not enable gain analysis")
    deck = resize_netlist(template, sizes, sizing_config, literal_wl=True)

    overrides = {
        str(element).lower(): str(line)
        for element, line in gain_config.get("source_overrides", {}).items()
    }
    found = set()
    lines = deck.splitlines()
    control_start = None
    control_end = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower() == ".control":
            control_start = index
        elif stripped.lower() == ".endc" and control_start is not None:
            control_end = index
            break
    if control_start is None or control_end is None:
        raise ValueError("template must contain a complete .control/.endc block")

    for index, line in enumerate(lines[:control_start]):
        stripped = line.strip()
        if not stripped or stripped.startswith("*") or stripped.startswith("."):
            continue
        element = stripped.split()[0].lower()
        if element in overrides:
            lines[index] = overrides[element]
            found.add(element)
    missing = set(overrides) - found
    if missing:
        raise ValueError(f"source override(s) not found in template: {', '.join(sorted(missing))}")

    control_block = [
        ".control",
        *[str(command) for command in gain_config.get("control", [])],
        f"wrdata {waveform_name} " + " ".join(gain_config.get("vectors", [])),
        ".endc",
    ]
    lines = [
        *lines[:control_start],
        *[str(line) for line in gain_config.get("extra_devices", [])],
        *control_block,
        *lines[control_end + 1:],
    ]
    return "\n".join(lines) + "\n"
