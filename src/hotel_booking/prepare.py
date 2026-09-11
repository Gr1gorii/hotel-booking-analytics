"""Download the authors' files and build a tidy booking table."""

import calendar
import hashlib
import io
import json
import re
import sqlite3
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6297060/supplementaryFiles"
CHECKSUMS = {
    "H1.csv": "6f38fda13794fd6cf6547d2432ffb79666200000eb45fd09e4b97fe49519a6ab",
    "H2.csv": "5107090ffdc5585a9e18299801e9d85c3b459c235542882183a502a73ef63c4d",
}


def download_data():
    folder = ROOT / "data/raw"
    folder.mkdir(parents=True, exist_ok=True)
    if not all((folder / name).exists() for name in CHECKSUMS):
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "HotelBookingPortfolio/0.1"})
        with urllib.request.urlopen(request, timeout=90) as response:
            archive = ZipFile(io.BytesIO(response.read()))
        csv_archive = ZipFile(io.BytesIO(archive.read("mmc2.zip")))
        # Read only the two data files, excluding the archive's macOS metadata.
        for name in CHECKSUMS:
            (folder / name).write_bytes(csv_archive.read(name))
    for name, expected in CHECKSUMS.items():
        actual = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {name}. Check the source before continuing.")


def snake_case(name):
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def clean_bookings(raw):
    df = raw.copy()
    df.columns = [snake_case(name) for name in df.columns]
    for name in df.select_dtypes(include="object"):
        df[name] = df[name].str.strip().replace({"NULL": np.nan, "": np.nan})
    # These are anonymous category codes, not quantities.
    for name in ["agent", "company"]:
        df[name] = df[name].fillna("Not recorded")
    df["country"] = df["country"].fillna("Unknown")
    df["meal"] = df["meal"].replace({"Undefined": "SC"})
    month_number = {name: number for number, name in enumerate(calendar.month_name) if name}
    df["arrival_date"] = pd.to_datetime(
        {
            "year": df["arrival_date_year"],
            "month": df["arrival_date_month"].map(month_number),
            "day": df["arrival_date_day_of_month"],
        },
        errors="raise",
    )
    df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"], errors="raise")
    df["arrival_month"] = df["arrival_date"].dt.month
    df["arrival_weekday"] = df["arrival_date"].dt.dayofweek
    df["prediction_date"] = df["arrival_date"] - pd.Timedelta(days=1)
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["no_guests"] = (df["adults"] + df["children"].fillna(0) + df["babies"]) == 0
    df["zero_nights"] = df["total_nights"].eq(0)
    df["adr_flag"] = (df["adr"] < 0) | (df["adr"] > 1000)
    df["adr_for_summary"] = df["adr"].mask(df["adr_flag"])
    df["lead_band"] = pd.cut(
        df["lead_time"],
        [-1, 7, 30, 90, 180, np.inf],
        labels=["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181+ days"],
    ).astype(str)
    df["days_before_arrival_canceled"] = (df["arrival_date"] - df["reservation_status_date"]).dt.days.where(
        df["is_canceled"].eq(1)
    )
    return df


def make_cohort(df):
    """Conservative late-cancellation cohort at the end of the day before arrival."""
    cutoff = df["arrival_date"] - pd.Timedelta(days=1)
    canceled = df["is_canceled"].eq(1)
    reason = pd.Series("Eligible", index=df.index)
    reason.loc[df["reservation_status_date"] < df["arrival_date"]] = "Inconsistent checkout date"
    reason.loc[canceled & (df["reservation_status_date"] < cutoff)] = "Already canceled before D-1"
    reason.loc[canceled & (df["reservation_status_date"] == cutoff)] = "Canceled on D-1, time unknown"
    reason.loc[df["lead_time"] < 2] = "Created on D-1 or arrival day"
    reason.loc[df["zero_nights"] | df["no_guests"]] = "No overnight stay or no guests"
    return df.loc[reason.eq("Eligible")].copy(), reason


def main():
    download_data()
    frames = []
    for filename, hotel in [("H1.csv", "Resort Hotel"), ("H2.csv", "City Hotel")]:
        part = pd.read_csv(ROOT / "data/raw" / filename)
        part["Hotel"] = hotel
        part["RowId"] = [f"{filename[:2]}:{i:05d}" for i in range(1, len(part) + 1)]
        frames.append(part)
    raw = pd.concat(frames, ignore_index=True)
    df = clean_bookings(raw)
    original_columns = [snake_case(name) for name in frames[0].columns if name != "RowId"]
    df["record_fingerprint"] = pd.util.hash_pandas_object(df[original_columns], index=False).astype(str)
    df["duplicate_record"] = df.duplicated(original_columns)
    cohort, reasons = make_cohort(df)
    df["model_cohort"] = reasons
    missing = raw.isna().sum()
    quality = {
        "raw_rows": len(df),
        "hotel_counts": df["hotel"].value_counts().to_dict(),
        "duplicate_rows": int(df["duplicate_record"].sum()),
        "zero_nights": int(df["zero_nights"].sum()),
        "no_guests": int(df["no_guests"].sum()),
        "adr_flagged": int(df["adr_flag"].sum()),
        "missing_before_cleaning": missing[missing > 0].to_dict(),
        "cohort_waterfall": reasons.value_counts().to_dict(),
        "eligible_rows": len(cohort),
        "eligible_cancellations": int(cohort["is_canceled"].sum()),
        "all_cancellations": int(df["is_canceled"].sum()),
        "source_url": SOURCE_URL,
        "sha256": CHECKSUMS,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    df.to_parquet(ROOT / "data/processed/bookings.parquet", index=False)
    with sqlite3.connect(ROOT / "artifacts/bookings.sqlite") as conn:
        df.to_sql("bookings", conn, if_exists="replace", index=False)
        conn.executescript((ROOT / "sql/analysis.sql").read_text())
        for name in [
            "hotel_summary",
            "monthly_demand",
            "channel_summary",
            "lead_time_summary",
            "cancellation_timing",
        ]:
            pd.read_sql_query(f"SELECT * FROM {name}", conn).to_csv(
                ROOT / f"artifacts/{name}.csv", index=False
            )
    (ROOT / "artifacts/data_quality.json").write_text(json.dumps(quality, indent=2))
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
