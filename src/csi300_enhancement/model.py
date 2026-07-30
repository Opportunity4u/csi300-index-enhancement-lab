from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
