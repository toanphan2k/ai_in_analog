#!/usr/bin/env python3
"""Newton-Raphson-based population search over a trained surrogate model."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from src.design_features import design_feature_names, design_feature_values
from src.selector import variable_names


def _sizes_from_normalized(
    normalized: np.ndarray,
    feature_names: Sequence[str],
    sizing_config: Mapping,
) -> dict[str, float]:
    bounds = sizing_config.get("optimization", {}).get("bounds", {})
    sizes = {}
    for name, value in zip(feature_names, normalized):
        bound = bounds[name]
        low, high = float(bound["min"]), float(bound["max"])
        if bound.get("scale", "linear") == "log":
            sizes[name] = low * (high / low) ** float(value)
        else:
            sizes[name] = low + float(value) * (high - low)
    return sizes


def _predict_targets(
    normalized: np.ndarray,
    sizing_config: Mapping,
    model: Mapping,
) -> dict[str, float]:
    sizes = _sizes_from_normalized(normalized, variable_names(sizing_config), sizing_config)
    values = design_feature_values(sizes, sizing_config)
    features = np.asarray([[values[name] for name in model["feature_names"]]], dtype=float)
    predictions = {}
    for target, target_model in model["models"].items():
        scaled_features = target_model["x_scaler"].transform(features)
        transformed = np.asarray(target_model["estimator"].predict(scaled_features)).reshape(-1, 1)
        target_scaler = target_model["y_scaler"]
        if target_scaler is not None:
            transformed = target_scaler.inverse_transform(transformed)
        value = float(transformed[0, 0])
        if model.get("target_transform") == "log-standard":
            value = float(np.exp(value))
        predictions[target] = value
    return predictions


def optimize_candidates(
    model: Mapping,
    sizing_config: Mapping,
    *,
    objective: str | None,
    maximize_objective: bool,
    delay_weight: float,
    speed_weight: float,
    power_weight: float,
    top_k: int,
    seed: int,
    iterations: int,
    population_size: int,
    max_power_w: float | None = None,
    min_valid_probability: float = 0.5,
) -> list[dict]:
    """Return ranked surrogate candidates; no simulator calls are made here."""
    configured_names = variable_names(sizing_config)
    feature_names = tuple(model.get("feature_names", ()))
    expected_features = design_feature_names(sizing_config)
    model_variables = tuple(model.get("design_variables", configured_names))
    if (
        set(model_variables) != set(configured_names)
        or len(model_variables) != len(configured_names)
    ):
        raise ValueError(
            "model design variables do not match optimization.variables in sizing.yaml"
        )
    allowed_features = {frozenset(expected_features), frozenset(configured_names)}
    if (
        frozenset(feature_names) not in allowed_features
        or len(set(feature_names)) != len(feature_names)
    ):
        raise ValueError("model features do not match configured raw and derived design features")
    if objective is not None and objective not in model.get("models", {}):
        raise ValueError(f"objective {objective!r} is not present in the model artifact")
    if objective is None and delay_weight > 0 and "delay_s" not in model.get("models", {}):
        raise ValueError(
            "weighted search requires a delay_s target when --delay-weight is positive"
        )
    if objective is None and power_weight > 0 and "power_w" not in model.get("models", {}):
        raise ValueError(
            "weighted search requires a power_w target when --power-weight is positive"
        )
    if objective is None and speed_weight > 0 and "speed_hz" not in model.get("models", {}):
        raise ValueError(
            "weighted search requires a speed_hz target when --speed-weight is positive"
        )
    if max_power_w is not None and "power_w" not in model.get("models", {}):
        raise ValueError("--max-power-w requires a power_w target in the model artifact")
    if not 0.0 <= min_valid_probability <= 1.0:
        raise ValueError("min_valid_probability must be between 0 and 1")
    if delay_weight < 0 or speed_weight < 0 or power_weight < 0:
        raise ValueError("objective weights must be nonnegative")
    if objective is None and delay_weight + speed_weight + power_weight == 0:
        raise ValueError("at least one objective weight must be positive")
    if top_k <= 0 or iterations <= 0 or population_size < 4:
        raise ValueError(
            "top_k and iterations must be positive; population_size must be at least 4"
        )

    dimension = len(configured_names)
    random = np.random.default_rng(seed)
    population = random.random((population_size, dimension))
    prediction_cache: dict[tuple[float, ...], dict[str, float]] = {}

    def predictions_at(normalized: np.ndarray) -> dict[str, float]:
        key = tuple(np.round(np.clip(normalized, 0.0, 1.0), 10))
        if key not in prediction_cache:
            predictions = _predict_targets(
                np.asarray(key), sizing_config, model
            )
            if "classifier" in model:
                sizes = _sizes_from_normalized(
                    np.asarray(key), configured_names, sizing_config
                )
                feature_values = design_feature_values(sizes, sizing_config)
                classifier = model["classifier"]
                classifier_x = classifier["scaler"].transform(
                    [[feature_values[name] for name in feature_names]]
                )
                classes = classifier["estimator"].classes_
                positive_class = np.flatnonzero(classes == 1)
                predictions["valid_probability"] = (
                    float(
                        classifier["estimator"].predict_proba(classifier_x)[
                            0, positive_class[0]
                        ]
                    )
                    if len(positive_class)
                    else 0.0
                )
            else:
                predictions["valid_probability"] = 1.0
            prediction_cache[key] = predictions
        return prediction_cache[key]

    def score(normalized: np.ndarray) -> float:
        predictions = predictions_at(normalized)
        if objective is not None:
            target_reference = model.get("target_reference", {}).get(objective, 1.0)
            objective_value = predictions[objective] / target_reference
            if maximize_objective:
                objective_value = -objective_value
        else:
            objective_value = 0.0
            if delay_weight:
                reference = model.get("target_reference", {}).get("delay_s", 1.0)
                objective_value += delay_weight * predictions["delay_s"] / reference
            if speed_weight:
                reference = model.get("target_reference", {}).get("speed_hz", 1.0)
                objective_value -= speed_weight * predictions["speed_hz"] / reference
            if power_weight:
                reference = model.get("target_reference", {}).get("power_w", 1.0)
                objective_value += power_weight * predictions["power_w"] / reference
        if max_power_w is None:
            power_penalty = 0.0
        else:
            violation = max(0.0, predictions["power_w"] / max_power_w - 1.0)
            power_penalty = 100.0 * violation**2
        classifier_penalty = 100.0 * max(
            0.0, min_valid_probability - predictions["valid_probability"]
        ) ** 2
        return objective_value + power_penalty + classifier_penalty

    finite_difference_step = 0.01

    for _ in range(iterations):
        population_scores = np.asarray([score(candidate) for candidate in population])
        order = np.argsort(population_scores)
        population = population[order]
        best = population[0].copy()
        elite_count = max(2, population_size // 5)
        elite = population[:elite_count]
        next_population = [best]

        while len(next_population) < population_size:
            current = population[len(next_population)].copy()
            if random.random() < 0.12:
                proposal = random.random(dimension)
            else:
                gradient = np.zeros(dimension)
                curvature = np.zeros(dimension)
                current_score = score(current)
                for axis in range(dimension):
                    positive = current.copy()
                    negative = current.copy()
                    positive[axis] = min(1.0, positive[axis] + finite_difference_step)
                    negative[axis] = max(0.0, negative[axis] - finite_difference_step)
                    positive_delta = positive[axis] - current[axis]
                    negative_delta = current[axis] - negative[axis]
                    positive_score = score(positive)
                    negative_score = score(negative)
                    span = positive_delta + negative_delta
                    gradient[axis] = (positive_score - negative_score) / max(span, 1e-12)
                    curvature[axis] = (
                        positive_score - 2.0 * current_score + negative_score
                    ) / max((span / 2.0) ** 2, 1e-12)

                newton_step = np.clip(
                    gradient
                    / np.where(
                        np.abs(curvature) < 1e-6,
                        np.where(curvature < 0, -1e-6, 1e-6),
                        curvature,
                    ),
                    -0.15,
                    0.15,
                )
                newton_candidate = np.clip(current - newton_step, 0.0, 1.0)
                first, second = random.choice(len(population), size=2, replace=False)
                guide = elite[random.integers(len(elite))]
                population_move = (
                    current
                    + random.random() * (guide - current)
                    + 0.2 * (population[first] - population[second])
                )
                proposal = 0.65 * newton_candidate + 0.35 * population_move
            next_population.append(np.clip(proposal, 0.0, 1.0))
        population = np.asarray(next_population)

    ranked = sorted(population, key=score)
    results = []
    seen = set()
    for normalized in ranked:
        sizes = _sizes_from_normalized(normalized, configured_names, sizing_config)
        signature = tuple(round(sizes[name], 10) for name in configured_names)
        if signature in seen:
            continue
        seen.add(signature)
        predictions = predictions_at(normalized)
        feasible = (max_power_w is None or predictions["power_w"] <= max_power_w) and (
            predictions["valid_probability"] >= min_valid_probability
        )
        results.append(
            {
                "size": sizes,
                "predictions": predictions,
                "objective": predictions[objective] if objective else score(normalized),
                "score": score(normalized),
                "feasible": feasible,
                "valid_probability": predictions["valid_probability"],
            }
        )
        if len(results) >= top_k:
            break
    return results
