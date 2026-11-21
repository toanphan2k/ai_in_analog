"""Render a unified resized SPICE deck from a design template and its sizing YAML."""
from __future__ import annotations

from typing import Mapping

from scripts.netlist_tools import resize_netlist


def render_sized_deck(
    template: str,
    sizes: Mapping[str, float],
    sizing_config: Mapping,
) -> str:
    """Render one unified SPICE deck with literal W/L values for SKY130 binned models."""
    return resize_netlist(template, sizes, sizing_config, literal_wl=True)
