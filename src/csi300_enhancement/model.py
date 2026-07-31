from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_model_as_of(
    panel: pd.DataFrame,
    features: list[str],
    rebalance_dates: pd.DatetimeIndex,
    prediction_date: pd.Timestamp | str,
    knowledge_date: pd.Timestamp | str | None = None,
    horizon: int = 5,
    ridge_alpha: float = 10.0,
    training_mode: str = "expanding",
    rolling_years: int = 3,
    min_train_rows: int = 5_000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Refit the fixed Ridge specification using only fully matured labels."""
    prediction_date = pd.Timestamp(prediction_date).normalize()
    knowledge_date = pd.Timestamp(knowledge_date or prediction_date).normalize()
    target_col = f"ForwardReturn{horizon}D"
    target_date_col = f"TargetDate{horizon}D"
    usable = panel.dropna(subset=features + [target_col, target_date_col]).copy()
    train = usable[
        (usable[target_date_col] < knowledge_date)
        & (usable["Date"].isin(rebalance_dates))
    ]
    if training_mode == "rolling":
        train = train[train["Date"] >= knowledge_date - pd.DateOffset(years=rolling_years)]
    if len(train) < min_train_rows:
        raise ValueError(f"insufficient mature training rows: {len(train)} < {min_train_rows}")

    today = panel[panel["Date"] == prediction_date].dropna(subset=features).copy()
    if today.empty:
        raise ValueError(f"no complete factor rows for prediction date {prediction_date.date()}")
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=ridge_alpha, fit_intercept=True)),
    ])
    model.fit(train[features], train[target_col])
    train_start = pd.Timestamp(train["Date"].min())
    train_end = pd.Timestamp(train[target_date_col].max())
    today["PredictedReturn"] = model.predict(today[features])
    today["DecisionDate"] = prediction_date
    today["ModelTrainStart"] = train_start
    today["ModelTrainEnd"] = train_end
    today["TrainingMode"] = training_mode
    predictions = today[[
        "DecisionDate", "Ticker", "PredictedReturn", "ModelTrainStart",
        "ModelTrainEnd", "TrainingMode",
    ]]
    coefficients = pd.DataFrame({
        "ModelDate": prediction_date,
        "Feature": features,
        "Coefficient": model.named_steps["ridge"].coef_,
        "TrainStart": train_start,
        "TrainEnd": train_end,
        "TrainingMode": training_mode,
    })
    metadata = {
        "PredictionDate": prediction_date,
        "KnowledgeDate": knowledge_date,
        "TrainStart": train_start,
        "TrainEnd": train_end,
        "TrainingRows": len(train),
        "PredictionRows": len(predictions),
    }
    return predictions.reset_index(drop=True), coefficients, metadata


def walk_forward_predictions(
    panel: pd.DataFrame,
    features: list[str],
    rebalance_dates: pd.DatetimeIndex,
    oos_start: str,
    horizon: int = 5,
    ridge_alpha: float = 10.0,
    training_mode: str = "expanding",
    rolling_years: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_col = f"ForwardReturn{horizon}D"
    target_date_col = f"TargetDate{horizon}D"
    usable = panel.dropna(subset=features + [target_col, target_date_col]).copy()
    prediction_rows: list[pd.DataFrame] = []
    coefficient_rows: list[dict] = []
    model: Pipeline | None = None
    fitted_month: tuple[int, int] | None = None
    train_start_used: pd.Timestamp | None = None
    train_end_used: pd.Timestamp | None = None

    for date in rebalance_dates[rebalance_dates >= pd.Timestamp(oos_start)]:
        month_key = (date.year, date.month)
        if model is None or month_key != fitted_month:
            train = usable[usable[target_date_col] < date]
            if training_mode == "rolling":
                train = train[train["Date"] >= date - pd.DateOffset(years=rolling_years)]
            # Use one observation per weekly decision date to reduce overlapping-label dependence.
            train = train[train["Date"].isin(rebalance_dates)]
            if len(train) < 5_000:
                continue
            model = Pipeline([
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=ridge_alpha, fit_intercept=True)),
            ])
            model.fit(train[features], train[target_col])
            fitted_month = month_key
            train_start_used = train["Date"].min()
            train_end_used = train[target_date_col].max()
            coefs = model.named_steps["ridge"].coef_
            for name, coef in zip(features, coefs):
                coefficient_rows.append({
                    "ModelDate": date, "Feature": name, "Coefficient": float(coef),
                    "TrainStart": train_start_used, "TrainEnd": train_end_used,
                    "TrainingMode": training_mode,
                })

        today = panel[panel["Date"] == date].dropna(subset=features).copy()
        if model is None or today.empty:
            continue
        today["PredictedReturn"] = model.predict(today[features])
        today["DecisionDate"] = date
        today["ModelTrainStart"] = train_start_used
        today["ModelTrainEnd"] = train_end_used
        today["TrainingMode"] = training_mode
        prediction_rows.append(today[[
            "DecisionDate", "Ticker", "PredictedReturn", "ModelTrainStart",
            "ModelTrainEnd", "TrainingMode",
        ]])
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    coefficients = pd.DataFrame(coefficient_rows)
    return predictions, coefficients
