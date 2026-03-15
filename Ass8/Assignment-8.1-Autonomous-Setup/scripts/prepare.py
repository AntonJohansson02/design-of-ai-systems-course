from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_arrays(source_data_dir: Path, output_dir: Path, val_size: float, random_state: int) -> dict[str, tuple[int, ...]]:
    train_df = pd.read_csv(source_data_dir / "train.csv")
    test_df = pd.read_csv(source_data_dir / "test.csv")

    y = train_df["SalePrice"].to_numpy(dtype=np.float64)
    train_features = train_df.drop(columns=["SalePrice"])
    test_features = test_df.copy()

    all_features = pd.concat([train_features, test_features], axis=0, ignore_index=True)
    numeric_cols = all_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in all_features.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", _make_ohe()),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    X_all = preprocessor.fit_transform(all_features)
    X_train_full = X_all[: len(train_features)]
    X_test = X_all[len(train_features) :]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y,
        test_size=val_size,
        random_state=random_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X_train.npy", np.asarray(X_train, dtype=np.float32))
    np.save(output_dir / "X_val.npy", np.asarray(X_val, dtype=np.float32))
    np.save(output_dir / "X_test.npy", np.asarray(X_test, dtype=np.float32))
    np.save(output_dir / "y_train.npy", np.asarray(y_train, dtype=np.float32))
    np.save(output_dir / "y_val.npy", np.asarray(y_val, dtype=np.float32))
    np.save(output_dir / "test_ids.npy", test_df["Id"].to_numpy(dtype=np.int32))

    feature_names = preprocessor.get_feature_names_out()
    np.save(output_dir / "feature_names.npy", feature_names)

    metadata = {
        "val_size": val_size,
        "random_state": random_state,
        "train_shape": list(np.asarray(X_train).shape),
        "val_shape": list(np.asarray(X_val).shape),
        "test_shape": list(np.asarray(X_test).shape),
        "num_features": int(np.asarray(X_train).shape[1]),
    }
    (output_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "X_train": tuple(np.asarray(X_train).shape),
        "X_val": tuple(np.asarray(X_val).shape),
        "X_test": tuple(np.asarray(X_test).shape),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    default_source = project_root.parent / "Assignment-8.0-Your-mini-project"
    parser = argparse.ArgumentParser(description="Prepare Ames Housing arrays for autonomous model search.")
    parser.add_argument("--source-data-dir", type=Path, default=default_source)
    parser.add_argument("--output-dir", type=Path, default=project_root / "data" / "prepared")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    shapes = build_arrays(args.source_data_dir, args.output_dir, args.val_size, args.random_state)
    print("Prepared arrays:")
    for name, shape in shapes.items():
        print(f"  {name}: {shape}")


if __name__ == "__main__":
    main()
