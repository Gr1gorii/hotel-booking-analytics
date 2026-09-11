"""Compare two modest models with a lead-time rule on a later arrival period."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hotel_booking.prepare import ROOT, make_cohort

NUMERIC = [
    "lead_time",
    "arrival_month",
    "arrival_weekday",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
]
CATEGORICAL = ["hotel", "meal", "market_segment", "distribution_channel", "reserved_room_type"]
FEATURES = NUMERIC + CATEGORICAL
VALIDATION_START = pd.Timestamp("2016-07-01")
TEST_START = pd.Timestamp("2017-01-01")


def split_cohort(cohort):
    # A training label must already be known on the day the model is frozen.
    train = cohort[
        (cohort.prediction_date < VALIDATION_START) & (cohort.reservation_status_date < VALIDATION_START)
    ].copy()
    validation = cohort[
        (cohort.prediction_date >= VALIDATION_START)
        & (cohort.prediction_date < TEST_START)
        & (cohort.reservation_status_date < TEST_START)
    ].copy()
    test = cohort[cohort.prediction_date >= TEST_START].copy()
    return train, validation, test


def make_pipeline(estimator, scale=False):
    steps = [("missing", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    preprocess = ColumnTransformer(
        [
            ("numeric", Pipeline(steps), NUMERIC),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    return Pipeline([("prepare", preprocess), ("model", estimator)])


def evaluate(y, probability, flags):
    tn, fp, fn, tp = confusion_matrix(y, flags, labels=[0, 1]).ravel()
    return {
        "rows": len(y),
        "cancellations": int(np.sum(y)),
        "prevalence": float(np.mean(y)),
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "review_rate": float(np.mean(flags)),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
    }


def bootstrap_ap(y, probability, dates, repeats=200):
    """Resample arrival dates together to allow some within-day dependence."""
    rng = np.random.default_rng(42)
    codes, unique = pd.factorize(dates)
    groups = [np.flatnonzero(codes == i) for i in range(len(unique))]
    values = []
    for _ in range(repeats):
        chosen = rng.integers(0, len(groups), len(groups))
        index = np.concatenate([groups[i] for i in chosen])
        if len(np.unique(y[index])) == 2:
            values.append(average_precision_score(y[index], probability[index]))
    return np.quantile(values, [0.025, 0.975]).tolist()


def main():
    bookings = pd.read_parquet(ROOT / "data/processed/bookings.parquet")
    cohort, _ = make_cohort(bookings)
    train, validation, test = split_cohort(cohort)
    models = {
        "Logistic regression": make_pipeline(LogisticRegression(max_iter=1500, C=1.0), scale=True),
        "Gradient boosting": make_pipeline(
            HistGradientBoostingClassifier(
                max_iter=160,
                max_leaf_nodes=15,
                learning_rate=0.06,
                min_samples_leaf=50,
                l2_regularization=2.0,
                early_stopping=False,
                random_state=42,
            )
        ),
    }
    validation_rows, thresholds = [], {}
    for name, model in models.items():
        model.fit(train[FEATURES], train.is_canceled)
        probability = model.predict_proba(validation[FEATURES])[:, 1]
        threshold = float(np.quantile(probability, 0.9))
        thresholds[name] = threshold
        validation_rows.append(
            {
                "model": name,
                "threshold": threshold,
                **evaluate(validation.is_canceled, probability, probability >= threshold),
            }
        )
        print(f"Fitted {name}", flush=True)
    winner = max(validation_rows, key=lambda row: row["average_precision"])["model"]

    # This rule is a binary risk score, not a calibrated probability.
    rule_score = test.lead_time.ge(90).astype(float).to_numpy()
    constant = np.full(len(test), train.is_canceled.mean())
    probabilities = {"Training prevalence": constant, "Lead time >= 90 days": rule_score}
    flags = {"Training prevalence": constant >= 0.5, "Lead time >= 90 days": test.lead_time.ge(90).to_numpy()}
    for name, model in models.items():
        probabilities[name] = model.predict_proba(test[FEATURES])[:, 1]
        flags[name] = probabilities[name] >= thresholds[name]
    rows = []
    for name, probability in probabilities.items():
        lower, upper = bootstrap_ap(test.is_canceled.to_numpy(), probability, test.arrival_date)
        metrics = evaluate(test.is_canceled, probability, flags[name])
        if name == "Lead time >= 90 days":
            metrics["brier_score"] = None
        rows.append({"model": name, **metrics, "ap_lower": lower, "ap_upper": upper})
    result = test[
        [
            "row_id",
            "arrival_date",
            "hotel",
            "market_segment",
            "lead_time",
            "lead_band",
            "total_nights",
            "is_canceled",
            "reservation_status",
            "record_fingerprint",
        ]
    ].copy()
    result["probability"] = probabilities[winner]
    result["flagged"] = flags[winner]
    result["rule_score"] = rule_score
    result["rule_flagged"] = flags["Lead time >= 90 days"]
    result["error"] = np.select(
        [
            result.flagged & result.is_canceled.eq(0),
            ~result.flagged & result.is_canceled.eq(1),
            result.flagged & result.is_canceled.eq(1),
        ],
        ["False positive", "False negative", "True positive"],
        default="True negative",
    )

    bins = pd.qcut(result.probability, 10, duplicates="drop")
    calibration = (
        result.groupby(bins, observed=True)
        .agg(
            mean_prediction=("probability", "mean"),
            observed_rate=("is_canceled", "mean"),
            rows=("row_id", "size"),
        )
        .reset_index(drop=True)
    )
    slices = []
    result["arrival_month"] = result.arrival_date.dt.strftime("%Y-%m")
    for dimension in ["hotel", "market_segment", "lead_band", "arrival_month"]:
        for value, part in result.groupby(dimension):
            tp = int(((part.is_canceled == 1) & part.flagged).sum())
            positives = int(part.is_canceled.sum())
            reviews = int(part.flagged.sum())
            slices.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "rows": len(part),
                    "cancellations": positives,
                    "reviewed": reviews,
                    "false_positive": int((part.error == "False positive").sum()),
                    "false_negative": int((part.error == "False negative").sum()),
                    "precision": tp / reviews if reviews else None,
                    "recall": tp / positives if positives else None,
                }
            )

    importance = permutation_importance(
        models[winner],
        test[FEATURES],
        test.is_canceled,
        scoring="average_precision",
        n_repeats=3,
        random_state=42,
    )
    feature_importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "ap_drop": importance.importances_mean,
            "repeat_std": importance.importances_std,
        }
    )
    # Sensitivity only: this does not establish that identical rows were duplicates.
    unique_test = result.drop_duplicates("record_fingerprint")
    sensitivity = evaluate(unique_test.is_canceled, unique_test.probability, unique_test.flagged)
    split_summary = []
    for name, part in [("Train", train), ("Validation", validation), ("Test", test)]:
        split_summary.append(
            {
                "split": name,
                "rows": len(part),
                "positives": int(part.is_canceled.sum()),
                "rate": float(part.is_canceled.mean()),
                "arrival_start": str(part.arrival_date.min().date()),
                "arrival_end": str(part.arrival_date.max().date()),
                "prediction_start": str(part.prediction_date.min().date()),
                "prediction_end": str(part.prediction_date.max().date()),
            }
        )
    metadata = {
        "selected_model": winner,
        "threshold": thresholds[winner],
        "features": FEATURES,
        "selection_metric": "Validation average precision",
        "splits": split_summary,
        "eligible_rows": len(cohort),
        "labels_not_yet_available_excluded": len(cohort) - len(train) - len(validation) - len(test),
        "unique_test_sensitivity": sensitivity,
        "interval_method": "200 bootstrap samples of arrival dates, 95% percentile interval; two hotels only",
    }
    folder = ROOT / "artifacts"
    pd.DataFrame(rows).to_csv(folder / "model_metrics.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(folder / "validation_metrics.csv", index=False)
    calibration.to_csv(folder / "calibration.csv", index=False)
    pd.DataFrame(slices).to_csv(folder / "error_slices.csv", index=False)
    feature_importance.sort_values("ap_drop", ascending=False).to_csv(
        folder / "feature_importance.csv", index=False
    )
    result.to_parquet(folder / "test_predictions.parquet", index=False)
    examples = pd.concat(
        [
            result[result.error == "False positive"].nlargest(5, "probability"),
            result[result.error == "False negative"].nsmallest(5, "probability"),
        ]
    )
    examples.drop(columns="record_fingerprint").to_csv(folder / "error_examples.csv", index=False)
    (folder / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    joblib.dump(
        {"pipeline": models[winner], "features": FEATURES, "threshold": thresholds[winner]},
        folder / "cancellation_model.joblib",
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
