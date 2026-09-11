"""Write the short business readout from computed results."""

import calendar
import json

import pandas as pd

from hotel_booking.prepare import ROOT


def main():
    df = pd.read_parquet(ROOT / "data/processed/bookings.parquet")
    hotel = df.groupby("hotel").is_canceled.mean()
    segment = df.groupby("market_segment").is_canceled.mean()
    lead = df.groupby("lead_band").is_canceled.mean()
    monthly = df[df.arrival_date.dt.year == 2016].groupby(["hotel", "arrival_month"]).size()
    peaks = {name: int(monthly.loc[name].idxmax()) for name in hotel.index}
    canceled = df[df.is_canceled == 1]
    median_days = canceled.days_before_arrival_canceled.median()
    early_share = canceled.days_before_arrival_canceled.gt(7).mean()
    brief = f"""# Five findings for a hotel manager

Scope: all {len(df):,} source bookings, July 2015 to August 2017. Identical records are retained.
These are observations from two Portuguese hotels, followed by proposed actions to test.

1. **The city hotel had more cancellations.** Its cancellation rate was {hotel["City Hotel"]:.1%},
   compared with {hotel["Resort Hotel"]:.1%} at the resort. Use separate hotel baselines in planning.
   Do not interpret the difference as an effect of hotel type; there is only one hotel of each type.

2. **October led the complete-year arrival counts.** In 2016, the only complete year, the city hotel's largest
   scheduled-arrival month was {calendar.month_name[peaks["City Hotel"]]}; the resort's was also
   {calendar.month_name[peaks["Resort Hotel"]]}. Build separate monthly planning views and compare
   gross bookings with completed stays. Booking counts alone do not establish staffing needs.

3. **Channel mix deserves attention.** Online TA bookings had a {segment["Online TA"]:.1%}
   cancellation rate, direct bookings {segment["Direct"]:.1%}, and group bookings {segment["Groups"]:.1%}.
   Review group release dates and channel terms, then compare similar lead-time and hotel cohorts.
   Do not shift budget based only on these unadjusted rates; commissions and acquisition costs are absent.

4. **Long-lead bookings carry more cancellation uncertainty overall.** The rate was
   {lead["181+ days"]:.1%} beyond 180 days, versus {lead["0-7 days"]:.1%} at 0-7 days.
   Track booking pickup and cancellations by lead-time band. This pattern does not automatically
   survive in the D-1 model cohort, where most earlier cancellations have already been removed.

5. **Most cancellations are known before the last week.** The median recorded cancellation was
   {median_days:.0f} days before arrival; {early_share:.1%} were recorded more than seven days before.
   An earlier confirmation workflow may be more useful than a last-day score. Test that idea with
   timestamped booking snapshots, contact outcomes and costs. The current table cannot measure its effect.

The data do not establish occupancy, RevPAR, recoverable revenue or the effect of any intervention.
Revalidate the definitions and all recommendations with a Milan hotel's current PMS data.

Source: [Antonio, de Almeida and Nunes (2019)](https://doi.org/10.1016/j.dib.2018.11.126).
Calculations: `sql/analysis.sql` and `src/hotel_booking/report.py`.
"""
    (ROOT / "reports/management_brief.md").write_text(brief)
    metrics = pd.read_csv(ROOT / "artifacts/model_metrics.csv").set_index("model")
    meta = json.loads((ROOT / "artifacts/model_metadata.json").read_text())
    chosen = metrics.loc[meta["selected_model"]]
    baseline = metrics.loc["Training prevalence"]
    rule = metrics.loc["Lead time >= 90 days"]
    quality = json.loads((ROOT / "artifacts/data_quality.json").read_text())
    unique = meta["unique_test_sensitivity"]
    slices = pd.read_csv(ROOT / "artifacts/error_slices.csv")
    city_errors = slices[(slices.dimension == "hotel") & (slices.value == "City Hotel")].iloc[0]
    resort_errors = slices[(slices.dimension == "hotel") & (slices.value == "Resort Hotel")].iloc[0]
    readout = f"""# Model results

{meta["selected_model"]} was selected using validation average precision, before the test evaluation.
The test set contains {int(chosen.rows):,} bookings and {int(chosen.cancellations):,} late cancellations
or no-shows ({chosen.prevalence:.2%}).

| Method | Average precision | ROC AUC | Brier score | Precision | Recall | Review share |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for name, row in metrics.iterrows():
        brier = f"{row.brier_score:.5f}" if pd.notna(row.brier_score) else "n/a"
        readout += f"| {name} | {row.average_precision:.4f} | {row.roc_auc:.4f} | {brier} | {row.precision:.2%} | {row.recall:.1%} | {row.review_rate:.1%} |\n"
    readout += f"""
The selected model's AP is {chosen.average_precision:.4f}, with a 95% arrival-date bootstrap interval
of {chosen.ap_lower:.4f} to {chosen.ap_upper:.4f}. The prevalence baseline has AP {baseline.average_precision:.4f}.
This is a small ranking improvement, not a deployment result. The model's Brier score
({chosen.brier_score:.5f}) is worse than the constant baseline ({baseline.brier_score:.5f}).

At the frozen threshold ({meta["threshold"]:.5f}), the selected model flags
{int(chosen.true_positive):,} actual cancellations and creates {int(chosen.false_positive):,} false alarms.
It misses {int(chosen.false_negative):,} cancellations. The fixed rule flags
{int(rule.true_positive):,} cancellations with {int(rule.false_positive):,} false alarms.
These precision and recall figures have different workloads; they are not an equal-capacity experiment.

The learned threshold was set to flag about 10% of validation bookings, but flags
{chosen.review_rate:.1%} on test. This is direct evidence that a fixed score threshold does not
preserve the workload over time. The dashboard lets a reader inspect the tradeoff without changing
the locked benchmark. Accuracy would reward predicting almost everyone as non-canceled and is omitted.

## Error analysis

The city hotel contributes {int(city_errors.false_positive):,} false alarms on
{int(city_errors.rows):,} test bookings. Its recall is {city_errors.recall:.1%} and precision
{city_errors.precision:.1%}. The resort has fewer false alarms ({int(resort_errors.false_positive):,}),
but recall falls to {resort_errors.recall:.1%}: {int(resort_errors.false_negative):,} of its
{int(resort_errors.cancellations):,} cancellations are missed. A single pooled threshold therefore
creates different service workloads and miss rates across the two hotels.

`artifacts/error_slices.csv` includes hotel, market segment, lead-time and monthly errors, with
denominators. `artifacts/error_examples.csv` contains five confident false alarms and five low-score
misses. These examples illustrate errors; they do not establish their cause.

The selected features capture booked-stay context but do not measure last-minute intent, travel
disruption or contact responses. These are possible missing signals, not verified explanations.
The observed score distribution shift warrants a shadow evaluation before any threshold is reused.

## Data sensitivity and limitations

There are {quality["duplicate_rows"]:,} identical source rows. Retaining only one identical record
in the test evaluation gives AP {unique["average_precision"]:.4f} on {unique["rows"]:,} rows.
This only reweights test evaluation. It does not retrain the model or prove which rows were duplicates.
Training/validation exclude {meta["labels_not_yet_available_excluded"]:,} eligible rows whose final
labels were not available before their respective evaluation freeze.

The model remains an offline case study. Two old Portuguese hotels, missing event timestamps and
unknown group dependence limit both the evaluation and transfer to Milan. The outcome is a late
cancellation/no-show proxy, not all cancellation risk at booking creation. No intervention value,
cost savings or revenue uplift was observed.
"""
    (ROOT / "reports/model_results.md").write_text(readout)
    print("Wrote management_brief.md and model_results.md")


if __name__ == "__main__":
    main()
