"""Parse and resize a SPICE netlist using a netlist-specific YAML configuration."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Mapping, Sequence

import yaml


def load_sizing_config(path: str | Path) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError(f"sizing config must contain a YAML mapping: {path}")
    if not isinstance(config.get("parameters"), dict) or not isinstance(config.get("groups"), dict):
        raise ValueError("sizing config must define 'parameters' and 'groups' mappings")
    return config


def fmt_um(value: float) -> str:
    value = float(value)
    if not 0 < value < 1e6:
        raise ValueError(f"W/L must be positive and in micrometres: {value}")
    return f"{value:.6g}"


def fmt_geometry(width_um: float, diff_len_um: float) -> Dict[str, float]:
    area = width_um * diff_len_um
    perimeter = 2.0 * (width_um + diff_len_um)
    return {"AD": area, "AS": area, "PD": perimeter, "PS": perimeter}


def _configured_devices(config: Mapping) -> Dict[str, tuple[str, str]]:
    parameters = config["parameters"]
    devices: Dict[str, tuple[str, str]] = {}
    for group_name, group in config["groups"].items():
        if not isinstance(group, dict):
            raise ValueError(f"group {group_name!r} must be a mapping")
        width_key, length_key = group.get("width"), group.get("length")
        if width_key not in parameters or length_key not in parameters:
            raise ValueError(f"group {group_name!r} refers to an undefined parameter")
        group_devices = group.get("devices")
        if not isinstance(group_devices, list) or not group_devices:
            raise ValueError(f"group {group_name!r} must list one or more devices")
        for label in group_devices:
            normalized = str(label).lower()
            if normalized in devices:
                raise ValueError(f"device {label!r} appears in more than one sizing group")
            devices[normalized] = (width_key, length_key)
    return devices


def _parameter_values(config: Mapping, sizes: Mapping[str, float]) -> Dict[str, float]:
    parameters = config["parameters"]
    unknown = set(sizes) - set(parameters)
    if unknown:
        raise ValueError(f"unknown size parameter(s): {', '.join(sorted(unknown))}")

    values: Dict[str, float] = {}
    for key, definition in parameters.items():
        if not isinstance(definition, dict) or not definition.get("spice_name"):
            raise ValueError(f"parameter {key!r} must define a spice_name")
        if key in sizes:
            value = sizes[key]
        elif "default" in definition:
            value = definition["default"]
        elif definition.get("required", False):
            raise KeyError(f"missing size for {key}")
        else:
            raise ValueError(f"parameter {key!r} needs either a size or a default")
        values[key] = float(value)
    return values


def _set_param_declaration(deck: str, spice_name: str, value: str) -> str:
    assignment = re.compile(
        rf"(?i)(?<![A-Za-z0-9_])({re.escape(spice_name)}\s*=\s*)[^\s]+"
    )
    lines = deck.splitlines(keepends=True)
    count = 0
    for index, line in enumerate(lines):
        if not re.match(r"(?i)^\s*\.param\b", line):
            continue
        lines[index], replacements = assignment.subn(
            lambda match: match.group(1) + value,
            line,
        )
        count += replacements
    if count != 1:
        raise ValueError(
            f"expected one .param declaration for {spice_name}, found {count}"
        )
    return "".join(lines)


def resize_netlist(
    template: str,
    sizes: Mapping[str, float],
    sizing_config: Mapping,
    *,
    literal_wl: bool = False,
) -> str:
    """Resize configured devices, keeping group relationships from YAML."""
    parameters = sizing_config["parameters"]
    values = _parameter_values(sizing_config, sizes)
    parameter_names = {
        key: str(definition["spice_name"])
        for key, definition in parameters.items()
    }

    rendered = template
    for key, value in values.items():
        rendered = _set_param_declaration(
            rendered,
            parameter_names[key],
            fmt_um(value),
        )

    geometry = sizing_config.get("geometry", {})
    diff_len = float(geometry.get("diffusion_length_um", 0.29))
    if diff_len <= 0:
        raise ValueError("geometry.diffusion_length_um must be positive")

    device_roles = _configured_devices(sizing_config)
    seen = set()
    lines = rendered.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("*") or stripped.startswith("."):
            continue
        label = stripped.split()[0]
        role = device_roles.get(label.lower())
        if role is None:
            continue

        width_key, length_key = role
        width = values[width_key]
        length = values[length_key]
        width_ref = parameter_names[width_key]
        length_ref = parameter_names[length_key]
        width_spice = fmt_um(width) if literal_wl else f"{{{width_ref}}}"
        length_spice = fmt_um(length) if literal_wl else f"{{{length_ref}}}"
        updated = re.sub(r"(?i)\bW\s*=\s*[^\s]+", f"W={width_spice}", stripped)
        updated = re.sub(r"(?i)\bL\s*=\s*[^\s]+", f"L={length_spice}", updated)
        for name, value in fmt_geometry(width, diff_len).items():
            updated = re.sub(
                rf"(?i)\b{name}\s*=\s*[^\s]+",
                f"{name}={value:.6g}",
                updated,
            )
        lines[i] = updated
        seen.add(label.lower())

    missing = set(device_roles) - seen
    if missing:
        raise ValueError(f"template is missing device instances: {', '.join(sorted(missing))}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--sizing-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes", required=True, help="JSON object of W/L values in micrometres")
    parser.add_argument("--wrdata", help="replace the generated transient output path")
    parser.add_argument(
        "--literal-wl",
        action="store_true",
        help="expand W/L on device cards for sky130 ngspice binned-model compatibility",
    )
    args = parser.parse_args(argv)

    template_path = Path(args.template)
    output_path = Path(args.output)
    sizes = json.loads(args.sizes)
    if not isinstance(sizes, dict):
        raise ValueError("--sizes must be a JSON object")

    rendered = resize_netlist(
        template_path.read_text(),
        sizes,
        load_sizing_config(args.sizing_config),
        literal_wl=args.literal_wl,
    )
    if args.wrdata:
        rendered = re.sub(
            r"(?m)^wrdata\s+\S+",
            f"wrdata {args.wrdata}",
            rendered,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
