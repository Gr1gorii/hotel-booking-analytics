import json
import sqlite3

import pandas as pd
import pytest

from hotel_booking.prepare import ROOT, make_cohort
from hotel_booking.train import FEATURES, TEST_START, VALIDATION_START, split_cohort


@pytest.fixture(scope="module")
def bookings():
    path = ROOT / "data/processed/bookings.parquet"
    if not path.exists():
        pytest.skip("Run python -m hotel_booking.prepare for source-data checks")
    return pd.read_parquet(path)


def test_reconciles_with_authors_counts(bookings):
    assert len(bookings) == 119390
    assert bookings.groupby("hotel").is_canceled.sum().to_dict() == {
        "City Hotel": 33102,
        "Resort Hotel": 11122,
    }
    assert bookings.row_id.is_unique
    assert bookings.children.isna().sum() == 4
    assert bookings.duplicate_record.sum() == 31994
    assert bookings.adr_flag.sum() == 2


def test_sql_matches_independent_counts(bookings):
    with sqlite3.connect(ROOT / "artifacts/bookings.sqlite") as conn:
        hotels = pd.read_sql_query("SELECT * FROM hotel_summary", conn)
        months = pd.read_sql_query("SELECT * FROM monthly_demand", conn)
        channels = pd.read_sql_query("SELECT * FROM channel_summary", conn)
    assert months.bookings.sum() == len(bookings)
    assert channels.cancellations.sum() == bookings.is_canceled.sum()
    assert (hotels.cancellation_rate * hotels.bookings).sum() == pytest.approx(44224)
    assert len(months) == 26 * 2


def test_prediction_cutoff_boundaries():
    df = pd.DataFrame(
        {
            "arrival_date": pd.to_datetime(["2017-02-10"] * 6),
            "reservation_status_date": pd.to_datetime(
                [
                    "2017-02-08",
                    "2017-02-09",
                    "2017-02-10",
                    "2017-02-12",
                    "2017-02-10",
                    "2017-02-12",
                ]
            ),
            "is_canceled": [1, 1, 1, 0, 1, 0],
            "lead_time": [20, 20, 20, 20, 1, 20],
            "zero_nights": [False] * 5 + [True],
            "no_guests": [False] * 6,
        }
    )
    cohort, reasons = make_cohort(df)
    assert cohort.index.tolist() == [2, 3]
    assert reasons.iloc[0] == "Already canceled before D-1"
    assert reasons.iloc[1] == "Canceled on D-1, time unknown"
    assert reasons.iloc[4] == "Created on D-1 or arrival day"


def test_temporal_split_and_label_availability(bookings):
    cohort, reasons = make_cohort(bookings)
    train, validation, test = split_cohort(cohort)
    assert train.prediction_date.max() < validation.prediction_date.min()
    assert validation.prediction_date.max() < test.prediction_date.min()
    assert train.reservation_status_date.max() < VALIDATION_START
    assert validation.reservation_status_date.max() < TEST_START
    assert test.prediction_date.min() >= TEST_START
    assert not set(train.row_id) & set(test.row_id)
    assert not set(train.record_fingerprint) & set(test.record_fingerprint)
    assert reasons.value_counts().sum() == len(bookings)
    assert len(cohort) == 67211
    canceled = cohort[cohort.is_canceled == 1]
    assert (canceled.reservation_status_date >= canceled.arrival_date).all()


def test_feature_boundary():
    forbidden = {
        "is_canceled",
        "reservation_status",
        "reservation_status_date",
        "booking_changes",
        "assigned_room_type",
        "deposit_type",
        "adr",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "record_fingerprint",
        "row_id",
        "model_cohort",
    }
    assert not set(FEATURES) & forbidden
    assert len(FEATURES) == 14


def test_saved_metrics_and_frozen_threshold():
    path = ROOT / "artifacts/test_predictions.parquet"
    if not path.exists():
        pytest.skip("Run python -m hotel_booking.train for model artifact checks")
    predictions = pd.read_parquet(path)
    meta = json.loads((ROOT / "artifacts/model_metadata.json").read_text())
    metrics = pd.read_csv(ROOT / "artifacts/model_metrics.csv").set_index("model")
    row = metrics.loc[meta["selected_model"]]
    assert predictions.probability.between(0, 1).all()
    assert (predictions.flagged == predictions.probability.ge(meta["threshold"])).all()
    assert row.true_positive + row.false_negative == predictions.is_canceled.sum()
    assert row.true_positive + row.false_positive == predictions.flagged.sum()
    assert metrics.loc["Training prevalence", "average_precision"] == pytest.approx(
        predictions.is_canceled.mean()
    )
    assert pd.isna(metrics.loc["Lead time >= 90 days", "brier_score"])
