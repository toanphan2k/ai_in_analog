"""Readers for ngspice transient, AC, noise-spectrum, and scalar metric files."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def load_transient(path: str | Path, default_names: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    """Load time plus single-scale transient vectors into a case-insensitive mapping."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        first = fh.readline().strip()
    if not first:
        raise ValueError(f"empty waveform file: {path}")

    fields = first.split()
    try:
        float(fields[0])
        header = None
        skiprows = 0
    except ValueError:
        header = fields[1:] if fields[0].startswith("#") else fields
        skiprows = 1

    data = np.loadtxt(path, comments="#", skiprows=skiprows, ndmin=2)
    if data.size == 0:
        raise ValueError(f"empty waveform file: {path}")
    if header is not None:
        names = [name.lower() for name in header]
        if len(names) != data.shape[1]:
            raise ValueError(
                f"waveform header has {len(names)} columns but data has {data.shape[1]}"
            )
    else:
        if len(default_names) != data.shape[1] - 1:
            raise ValueError("headerless transient file requires names for every vector")
        names = ["time", *(name.lower() for name in default_names)]

    if not names or names[0] not in {"time", "t"}:
        raise ValueError("transient waveform must have a time column first")
    return {name: data[:, i] for i, name in enumerate(names)}


def parse_ac(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Parse ngspice AC wrdata real/imaginary column pairs."""
    path = Path(path)
    lines = [
        line.split()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) < 2:
        raise ValueError(f"empty AC waveform file: {path}")

    header = lines[0]
    names = []
    for name in header[1:]:
        normalized = name.lower()
        if normalized not in names:
            names.append(normalized)
    data = np.asarray([[float(value) for value in row] for row in lines[1:]], dtype=float)
    if data.ndim != 2 or data.shape[1] != 1 + 2 * len(names):
        raise ValueError("AC wrdata must contain frequency and real/imaginary vector pairs")

    vectors = {}
    for index, name in enumerate(names):
        real = data[:, 1 + 2 * index]
        imag = data[:, 2 + 2 * index]
        vectors[name] = real + 1j * imag
    return data[:, 0], vectors


def load_noise_spectrum(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load frequency plus real noise spectral density vectors from ngspice wrdata."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        first = fh.readline().strip()
    if not first:
        raise ValueError(f"empty noise waveform file: {path}")

    fields = first.split()
    header = fields[1:] if fields[0].startswith("#") else fields
    data = np.loadtxt(path, comments="#", skiprows=1, ndmin=2)
    if data.size == 0:
        raise ValueError(f"empty noise waveform file: {path}")
    names = [name.lower() for name in header]
    if len(names) != data.shape[1] or names[0] not in {"frequency", "freq", "f"}:
        raise ValueError("noise wrdata must have frequency first followed by spectrum columns")
    return data[:, 0], {name: data[:, index] for index, name in enumerate(names[1:], start=1)}


def load_scalar_metrics(path: str | Path) -> dict[str, float]:
    """Parse space- or equals-separated scalar key/value lines written by ngspice."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"scalar metrics file does not exist: {path}")
    metrics: dict[str, float] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("*", "#")):
            continue
        cleaned = line.replace("=", " ")
        parts = cleaned.split()
        if len(parts) < 2:
            continue
        value = float(parts[1])
        if math.isfinite(value):
            metrics[parts[0].strip()] = value
    return metrics
