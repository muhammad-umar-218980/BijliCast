"""Leak-free preprocessing pipeline (Phase 3).

The saved pipeline contains three steps:

1. ``FeatureEngineer`` - stateless row-local feature engineering: calendar
   features from ``datetime`` and sine/cosine encoding of ``wind_direction``.
   No statistic is learned, so it cannot leak.
2. ``ColumnTransformer`` - median imputation + standardisation for numeric
   columns; mode imputation + one-hot encoding for categorical columns.
3. The estimator (added by :func:`build_pipeline`).

Every *learned* transformation is fitted on the training split only and is
re-fitted inside each cross-validation fold when the whole pipeline is passed
to ``GridSearchCV`` / ``RandomizedSearchCV``. Never pass pre-transformed arrays
to a CV search - pass the raw frame (``INPUT_COLUMNS``) and this pipeline.

Lag/rolling columns are deterministic, causal history features: they only look
backwards, use no fitted parameter, and are built identically at training and
inference time by ``src.features.build_features`` / ``src.features.inference``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (  # noqa: E402
    CATEGORICAL_FEATURES,
    INPUT_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    TIME_COL,
)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless, row-local feature engineering.

    Input : DataFrame with ``datetime`` and ``wind_direction``.
    Output: the same frame plus ``hour``, ``day_of_week``, ``month``,
    ``is_weekend``, ``wind_dir_sin``, ``wind_dir_cos``.
    """

    def fit(self, X, y=None):  # noqa: N803 - sklearn signature
        return self

    def transform(self, X):  # noqa: N803 - sklearn signature
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame")
        missing = [c for c in (TIME_COL, "wind_direction") if c not in X.columns]
        if missing:
            raise ValueError(f"FeatureEngineer missing columns: {missing}")

        df = X.copy()
        dt = pd.to_datetime(df[TIME_COL])
        df["hour"] = dt.dt.hour.astype("int8")
        df["day_of_week"] = dt.dt.dayofweek.astype("int8")
        df["month"] = dt.dt.month.astype("int8")
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype("int8")

        radians = np.deg2rad(df["wind_direction"].astype(float))
        df["wind_dir_sin"] = np.sin(radians)
        df["wind_dir_cos"] = np.cos(radians)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.asarray(INPUT_COLUMNS, dtype=object)


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Numeric: median impute -> standardise; categorical: mode -> one-hot."""
    numeric_features = list(numeric_features or NUMERIC_FEATURES)
    categorical_features = list(categorical_features or CATEGORICAL_FEATURES)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_pipeline(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """Feature engineering + imputation/scaling/encoding (no estimator)."""
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            (
                "preprocess",
                build_preprocessor(numeric_features, categorical_features),
            ),
        ]
    )


def build_pipeline(
    estimator,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """Full deployable pipeline: raw inputs -> features -> transforms -> model."""
    return Pipeline(
        steps=[
            ("prep", build_feature_pipeline(numeric_features, categorical_features)),
            ("model", estimator),
        ]
    )


def get_feature_names(pipeline) -> list[str]:
    """Readable output names after the preprocessing steps."""
    if isinstance(pipeline, Pipeline) and "preprocess" in pipeline.named_steps:
        pre = pipeline.named_steps["preprocess"]
    elif isinstance(pipeline, Pipeline) and "prep" in pipeline.named_steps:
        pre = pipeline.named_steps["prep"].named_steps["preprocess"]
    else:
        pre = pipeline
    return list(pre.get_feature_names_out())


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Raw pipeline input and label for a split (use this for CV searches)."""
    return df[INPUT_COLUMNS], df[TARGET].to_numpy()


def fit_on_train(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target: str = TARGET,
):
    """Fit the preprocessing pipeline on train and transform all three splits.

    Returns (fitted preprocessing Pipeline, X_train, y_train, X_val, y_val,
    X_test, y_test). For a quick inspection only; model training/CV should use
    :func:`build_pipeline` with the raw frames.
    """
    prep = build_feature_pipeline()
    X_train = prep.fit_transform(train[INPUT_COLUMNS])
    X_val = prep.transform(val[INPUT_COLUMNS])
    X_test = prep.transform(test[INPUT_COLUMNS])
    return (
        prep,
        X_train,
        train[target].to_numpy(),
        X_val,
        val[target].to_numpy(),
        X_test,
        test[target].to_numpy(),
    )


def smoke_test() -> None:
    """Quick self-check: scaler statistics must match the training split."""
    from src.utils.config import SPLITS_DIR

    if not (SPLITS_DIR / "train.parquet").exists():
        raise FileNotFoundError("run `python -m src.preprocessing.split` first")

    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val = pd.read_parquet(SPLITS_DIR / "val.parquet")
    test = pd.read_parquet(SPLITS_DIR / "test.parquet")

    prep, Xtr, ytr, Xva, yva, Xte, yte = fit_on_train(train, val, test)
    print("X_train", Xtr.shape, "| X_val", Xva.shape, "| X_test", Xte.shape)
    print("steps:", list(prep.named_steps))

    column_pre = prep.named_steps["preprocess"]
    scaler = column_pre.named_transformers_["num"].named_steps["scaler"]
    imputer = column_pre.named_transformers_["num"].named_steps["imputer"]

    engineered_train = prep.named_steps["features"].transform(train[INPUT_COLUMNS])
    expected = imputer.transform(engineered_train[NUMERIC_FEATURES]).mean(axis=0)
    if not np.allclose(scaler.mean_, expected, atol=1e-8):
        raise AssertionError("scaler was not fitted on the training split only")

    engineered_all = prep.named_steps["features"].transform(
        pd.concat([train, val, test])[INPUT_COLUMNS]
    )
    full_mean = np.nanmean(engineered_all[NUMERIC_FEATURES].to_numpy(), axis=0)
    drift = float(np.max(np.abs(scaler.mean_ - full_mean)))
    print(f"scaler mean == imputed train mean: OK | max drift vs full-data mean: {drift:.4f}")

    calendar = prep.named_steps["features"].transform(train[INPUT_COLUMNS].head(3))
    print("FeatureEngineer added:", [c for c in ("hour", "is_weekend", "wind_dir_sin") if c in calendar])
    print("feature names:", len(get_feature_names(prep)))


if __name__ == "__main__":
    smoke_test()
