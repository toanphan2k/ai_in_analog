#!/usr/bin/env python3
"""Train and evaluate surrogate models from a collected CSV or Parquet dataset."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Parquet input requires pandas and a parquet engine such as pyarrow"
            ) from exc
        return pd.read_parquet(path).to_dict(orient="records")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _make_regressor(method: str, seed: int, *, train_dir: Path | None = None):
    if method in {"gpr", "two-stage"}:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern

        return GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(1), length_scale_bounds=(1e-2, 1e2), nu=2.5),
            normalize_y=False,
            n_restarts_optimizer=2,
            random_state=seed,
        )
    if method == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError("LightGBM method requires the optional 'lightgbm' package") from exc
        return LGBMRegressor(n_estimators=300, learning_rate=0.04, random_state=seed, verbosity=-1)
    if method == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise RuntimeError("CatBoost method requires the optional 'catboost' package") from exc
        regressor_options = {
            "iterations": 300,
            "learning_rate": 0.04,
            "depth": 6,
            "random_seed": seed,
            "verbose": False,
        }
        if train_dir is not None:
            train_dir.mkdir(parents=True, exist_ok=True)
            regressor_options["train_dir"] = str(train_dir)
        return CatBoostRegressor(**regressor_options)
    if method == "mlp":
        from sklearn.neural_network import MLPRegressor

        return MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=2000,
            early_stopping=True,
            random_state=seed,
        )
    raise ValueError(f"unknown method {method!r}")


def _load_training_dependencies():
    try:
        from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Training requires scikit-learn; install project requirements first"
        ) from exc
    return accuracy_score, mean_absolute_error, r2_score, train_test_split, StandardScaler


def train(
    dataset: Path,
    output: Path,
    *,
    method: str,
    targets: Sequence[str],
    test_size: float,
    seed: int,
    target_transform: str,
    include_partial: bool = False,
) -> dict:
    """Fit per-target surrogates, evaluate held-out metrics, and save artifacts."""
    accuracy_score, mean_absolute_error, r2_score, train_test_split, StandardScaler = (
        _load_training_dependencies()
    )
    rows = _read_rows(dataset)
    if not rows:
        raise ValueError(f"dataset contains no rows: {dataset}")
    input_variables = [
        key.removeprefix("size__")
        for key in rows[0]
        if key.startswith("size__")
    ]
    feature_columns = [key for key in rows[0] if key.startswith("feature__")]
    if not feature_columns:
        feature_columns = [f"size__{name}" for name in input_variables]
    if not feature_columns or not input_variables:
        raise ValueError("dataset must contain size__ inputs and feature__ model features")
    features = [
        column.removeprefix("feature__")
        if column.startswith("feature__")
        else column.removeprefix("size__")
        for column in feature_columns
    ]

    def values_for(row):
        return [_number(row.get(column)) for column in feature_columns]

    usable_rows = [row for row in rows if all(value is not None for value in values_for(row))]
    if len(usable_rows) < 4:
        raise ValueError("at least four rows with complete sizing features are required")
    x_all = np.asarray([values_for(row) for row in usable_rows], dtype=float)
    indices = np.arange(len(usable_rows))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=seed
    )

    artifacts = {
        "format_version": 1,
        "method": method,
        "feature_names": features,
        "design_variables": input_variables,
        "target_names": list(targets),
        "target_transform": target_transform,
        "models": {},
        "target_reference": {},
    }
    status_counts = {}
    for row in usable_rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    evaluation = {
        "dataset": str(dataset),
        "method": method,
        "rows": len(usable_rows),
        "status_counts": status_counts,
        "partial_rows_included": include_partial,
        "targets": {},
    }

    if method == "two-stage":
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier

        labels = np.asarray([row.get("status") == "evaluated" for row in usable_rows], dtype=int)
        classifier_scaler = StandardScaler().fit(x_all[train_indices])
        if len(np.unique(labels[train_indices])) < 2:
            classifier = DummyClassifier(
                strategy="constant",
                constant=int(labels[train_indices][0]),
            )
        else:
            classifier = RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=seed
            )
        classifier.fit(classifier_scaler.transform(x_all[train_indices]), labels[train_indices])
        artifacts["classifier"] = {"scaler": classifier_scaler, "estimator": classifier}
        evaluation["classification_accuracy"] = float(
            accuracy_score(
                labels[test_indices],
                classifier.predict(classifier_scaler.transform(x_all[test_indices])),
            )
        )

    for target in targets:
        y_all = np.asarray([_number(row.get(target)) for row in usable_rows], dtype=object)
        valid = np.asarray([value is not None for value in y_all], dtype=bool)
        valid_statuses = {"evaluated", "partial"} if include_partial else {"evaluated"}
        valid &= np.asarray([row.get("status") in valid_statuses for row in usable_rows])
        valid_indices = np.flatnonzero(valid)
        target_train = np.intersect1d(train_indices, valid_indices)
        target_test = np.intersect1d(test_indices, valid_indices)
        if len(target_train) < 3 or len(target_test) < 1:
            raise ValueError(f"target {target!r} has too few usable train/test values")
        y = np.asarray([float(y_all[index]) for index in valid_indices], dtype=float)
        if target_transform == "log-standard" and np.any(y <= 0):
            raise ValueError(f"target {target!r} must be positive for log-standard scaling")
        transform_values = np.log(y) if target_transform == "log-standard" else y
        position_by_index = {int(index): position for position, index in enumerate(valid_indices)}
        y_train_raw = np.asarray(
            [transform_values[position_by_index[int(index)]] for index in target_train]
        )
        y_test_raw = np.asarray([transform_values[position_by_index[int(i)]] for i in target_test])

        x_scaler = StandardScaler().fit(x_all[target_train])
        y_scaler = (
            StandardScaler().fit(y_train_raw.reshape(-1, 1))
            if target_transform != "none"
            else None
        )
        estimator = _make_regressor(
            method,
            seed,
            train_dir=output.parent / "catboost_info" / target,
        )
        if method in {"gpr", "two-stage"}:
            from sklearn.gaussian_process.kernels import ConstantKernel, Matern

            estimator.kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=np.ones(len(features)),
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5,
            )
        train_x = x_scaler.transform(x_all[target_train])
        test_x = x_scaler.transform(x_all[target_test])
        train_y = (
            y_scaler.transform(y_train_raw.reshape(-1, 1)).ravel()
            if y_scaler is not None
            else y_train_raw
        )
        estimator.fit(train_x, train_y)
        predicted_scaled = estimator.predict(test_x).reshape(-1, 1)
        predicted_transformed = (
            y_scaler.inverse_transform(predicted_scaled).ravel()
            if y_scaler is not None
            else predicted_scaled.ravel()
        )
        if target_transform == "log-standard":
            predicted = np.exp(predicted_transformed)
            actual = np.exp(y_test_raw)
        else:
            predicted = predicted_transformed
            actual = y_test_raw
        target_metrics = {
            "train_rows": int(len(target_train)),
            "test_rows": int(len(target_test)),
            "mae": float(mean_absolute_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
            "mape_percent": float(
                100.0
                * np.mean(
                    np.abs(actual - predicted)
                    / np.maximum(np.abs(actual), np.finfo(float).tiny)
                )
            ),
        }
        importance: dict[str, dict[str, float] | dict[str, float]] = {}
        if method == "gpr":
            length_scales = np.asarray(estimator.kernel_.k2.length_scale, dtype=float)
            inverse_scales = 1.0 / np.maximum(length_scales, 1e-12)
            relative_importance = inverse_scales / inverse_scales.sum()
            importance["ard_length_scale"] = {
                name: float(value) for name, value in zip(features, length_scales)
            }
            importance["ard_relative_importance"] = {
                name: float(value) for name, value in zip(features, relative_importance)
            }
        if len(target_test) > 1:
            from sklearn.inspection import permutation_importance

            permutation = permutation_importance(
                estimator,
                test_x,
                y_scaler.transform(y_test_raw.reshape(-1, 1)).ravel()
                if y_scaler is not None
                else y_test_raw,
                scoring="neg_mean_absolute_error",
                n_repeats=5,
                random_state=seed,
            )
            importance["permutation"] = {
                name: float(value)
                for name, value in zip(features, permutation.importances_mean)
            }
        target_metrics["feature_importance"] = importance
        evaluation["targets"][target] = target_metrics
        target_reference = (
            float(np.exp(np.median(y_train_raw)))
            if target_transform == "log-standard"
            else float(np.median(y_train_raw))
        )
        artifacts["target_reference"][target] = max(abs(target_reference), 1e-12)
        artifacts["models"][target] = {
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "estimator": estimator,
        }

    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "Saving models requires joblib, normally installed with scikit-learn"
        ) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, output)
    evaluation_path = output.with_name(output.stem + "_evaluation.json")
    evaluation_path.write_text(
        json.dumps(evaluation, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return evaluation


def main(argv: Sequence[str] | None = None) -> int:
    """Parse training options and report held-out surrogate performance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="input CSV or Parquet dataset")
    parser.add_argument(
        "--method",
        choices=("gpr", "lightgbm", "catboost", "mlp", "two-stage"),
        default="gpr",
    )
    parser.add_argument(
        "--targets",
        default="delay_s,power_w",
        help="comma-separated metric columns",
    )
    parser.add_argument("--output", default="model.pkl")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target-transform",
        choices=("standard", "log-standard", "none"),
        default="log-standard",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="include partial simulation results in regression; failed rows remain excluded",
    )
    args = parser.parse_args(argv)
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    if not targets:
        parser.error("--targets must name at least one metric column")
    dataset = Path(args.dataset)
    if not dataset.is_absolute():
        dataset = PROJECT_DIR / dataset
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_DIR / output
    report = train(
        dataset,
        output,
        method=args.method,
        targets=targets,
        test_size=args.test_size,
        seed=args.seed,
        target_transform=args.target_transform,
        include_partial=args.include_partial,
    )
    print(json.dumps({"model": str(output), "evaluation": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
