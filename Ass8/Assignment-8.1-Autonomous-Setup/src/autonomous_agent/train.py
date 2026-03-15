from __future__ import annotations

import json
import math
import os
import time
import traceback
from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from xgboost import XGBRegressor

HYPERPARAM_SPECS: dict[str, dict[str, Any]] = {
    "n_estimators": {"type": "int", "min": 50, "max": 2000, "default": [1000]},
    "max_depth": {"type": "int", "min": 3, "max": 10, "default": [4]},
    "learning_rate": {"type": "float", "min": 0.01, "max": 0.3, "default": [0.05]},
    "subsample": {"type": "float", "min": 0.5, "max": 1.0, "default": [1.0]},
    "colsample_bytree": {"type": "float", "min": 0.5, "max": 1.0, "default": [1.0]},
    "gamma": {"type": "float", "min": 0.0, "max": 5.0, "default": [0.0]},
    "reg_alpha": {"type": "float", "min": 0.0, "max": 5.0, "default": [0.0]},
    "reg_lambda": {"type": "float", "min": 0.0, "max": 5.0, "default": [1.0]},
}

TOGGLE_DEFAULTS: dict[str, bool] = {
    "add_squared_features": False,
    "use_hist_tree_method": False,
}

MAX_VALUES_PER_PARAM = 5
MAX_GRID_SIZE = 128


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def _emit_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(_to_builtin(payload), sort_keys=True))


def _coerce_value(key: str, raw: Any, spec: dict[str, Any]) -> float | int:
    if isinstance(raw, bool):
        raise ValueError(f"{key} cannot use bool values")

    expected = spec["type"]
    if expected == "int":
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError(f"{key} must use integer values")
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} has non-integer value: {raw!r}") from exc
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} has non-float value: {raw!r}") from exc

    lower = spec["min"]
    upper = spec["max"]
    value = min(max(value, lower), upper)
    return value


def _normalize_switchboard(raw_payload: dict[str, Any]) -> tuple[dict[str, bool], dict[str, list[float | int]], int]:
    allowed = set(TOGGLE_DEFAULTS) | set(HYPERPARAM_SPECS)
    unknown = sorted(set(raw_payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys: {', '.join(unknown)}")

    toggles: dict[str, bool] = {}
    for key, default in TOGGLE_DEFAULTS.items():
        raw_value = raw_payload.get(key, default)
        if not isinstance(raw_value, bool):
            raise ValueError(f"Toggle '{key}' must be a boolean")
        toggles[key] = raw_value

    param_grid: dict[str, list[float | int]] = {}
    grid_size = 1
    for key, spec in HYPERPARAM_SPECS.items():
        raw_value = raw_payload.get(key, spec["default"])
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        if not values:
            raise ValueError(f"{key} list cannot be empty")
        if len(values) > MAX_VALUES_PER_PARAM:
            raise ValueError(f"{key} list exceeds max length {MAX_VALUES_PER_PARAM}")

        normalized_values = sorted({_coerce_value(key, value, spec) for value in values})
        if not normalized_values:
            raise ValueError(f"{key} list had no usable values")
        param_grid[key] = normalized_values
        grid_size *= len(normalized_values)

    if grid_size > MAX_GRID_SIZE:
        raise ValueError(f"Grid size {grid_size} exceeds cap {MAX_GRID_SIZE}")
    return toggles, param_grid, grid_size


def _load_switchboard_payload() -> dict[str, Any]:
    if not os.path.exists("hyperparams.json"):
        return {}
    with open("hyperparams.json", "r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError("hyperparams.json must contain a JSON object")
    return payload


def _apply_feature_toggles(X_train: np.ndarray, X_val: np.ndarray, toggles: dict[str, bool]) -> tuple[np.ndarray, np.ndarray]:
    train_arr = np.asarray(X_train, dtype=np.float32)
    val_arr = np.asarray(X_val, dtype=np.float32)

    if toggles["add_squared_features"]:
        train_sq = np.square(train_arr)
        val_sq = np.square(val_arr)
        train_arr = np.concatenate([train_arr, train_sq], axis=1)
        val_arr = np.concatenate([val_arr, val_sq], axis=1)

    return train_arr, val_arr


def _build_predefined_split(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, PredefinedSplit]:
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    test_fold = np.concatenate(
        [
            np.full(X_train.shape[0], -1, dtype=np.int32),
            np.zeros(X_val.shape[0], dtype=np.int32),
        ]
    )
    cv = PredefinedSplit(test_fold=test_fold)
    return X_all, y_all, cv


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> int:
    started_at = time.perf_counter()
    try:
        X_train = np.load("X_train.npy")
        y_train = np.load("y_train.npy")
        X_val = np.load("X_val.npy")
        y_val = np.load("y_val.npy")

        payload = _load_switchboard_payload()
        toggles, param_grid, grid_size = _normalize_switchboard(payload)

        X_train_t, X_val_t = _apply_feature_toggles(X_train, X_val, toggles)
        X_all, y_all, cv = _build_predefined_split(X_train_t, X_val_t, y_train, y_val)

        base_params: dict[str, Any] = {
            "random_state": 42,
            "objective": "reg:squarederror",
            "verbosity": 0,
            "n_jobs": 1,
        }
        if toggles["use_hist_tree_method"]:
            base_params["tree_method"] = "hist"

        estimator = XGBRegressor(**base_params)
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            refit=False,
            n_jobs=1,
            error_score="raise",
            verbose=0,
        )

        search.fit(X_all, y_all)

        best_score = float(-search.best_score_)
        best_params = {k: _to_builtin(v) for k, v in search.best_params_.items()}
        elapsed_s = time.perf_counter() - started_at

        _emit_payload(
            {
                "status": "ok",
                "score": best_score,
                "metric": "rmse",
                "grid_size": grid_size,
                "best_params": best_params,
                "toggles": toggles,
                "elapsed_seconds": round(elapsed_s, 4),
                "val_rmse": best_score,
            }
        )
        return 0
    except ValueError as exc:
        elapsed_s = time.perf_counter() - started_at
        _emit_payload(
            {
                "status": "config_error",
                "error_type": "ValueError",
                "message": str(exc),
                "elapsed_seconds": round(elapsed_s, 4),
            }
        )
        return 2
    except Exception as exc:  # pragma: no cover - runtime safety net
        elapsed_s = time.perf_counter() - started_at
        _emit_payload(
            {
                "status": "runtime_error",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "elapsed_seconds": round(elapsed_s, 4),
                "traceback_tail": " | ".join(traceback.format_exc().strip().splitlines()[-3:]),
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
