# Model results

Logistic regression was selected using validation average precision, before the test evaluation.
The test set contains 22,639 bookings and 468 late cancellations
or no-shows (2.07%).

| Method | Average precision | ROC AUC | Brier score | Precision | Recall | Review share |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training prevalence | 0.0207 | 0.5000 | 0.02028 | 0.00% | 0.0% | 0.0% |
| Lead time >= 90 days | 0.0195 | 0.4661 | n/a | 1.76% | 37.6% | 44.3% |
| Logistic regression | 0.0271 | 0.5769 | 0.02080 | 2.53% | 30.1% | 24.6% |
| Gradient boosting | 0.0267 | 0.5770 | 0.02171 | 2.50% | 10.7% | 8.8% |

The selected model's AP is 0.0271, with a 95% arrival-date bootstrap interval
of 0.0230 to 0.0338. The prevalence baseline has AP 0.0207.
This is a small ranking improvement, not a deployment result. The model's Brier score
(0.02080) is worse than the constant baseline (0.02028).

At the frozen threshold (0.03847), the selected model flags
141 actual cancellations and creates 5,434 false alarms.
It misses 327 cancellations. The fixed rule flags
176 cancellations with 9,843 false alarms.
These precision and recall figures have different workloads; they are not an equal-capacity experiment.

The learned threshold was set to flag about 10% of validation bookings, but flags
24.6% on test. This is direct evidence that a fixed score threshold does not
preserve the workload over time. The dashboard lets a reader inspect the tradeoff without changing
the locked benchmark. Accuracy would reward predicting almost everyone as non-canceled and is omitted.

## Error analysis

The city hotel contributes 4,807 false alarms on
14,810 test bookings. Its recall is 36.7% and precision
2.3%. The resort has fewer false alarms (627),
but recall falls to 17.8%: 134 of its
163 cancellations are missed. A single pooled threshold therefore
creates different service workloads and miss rates across the two hotels.

`artifacts/error_slices.csv` includes hotel, market segment, lead-time and monthly errors, with
denominators. `artifacts/error_examples.csv` contains five confident false alarms and five low-score
misses. These examples illustrate errors; they do not establish their cause.

The selected features capture booked-stay context but do not measure last-minute intent, travel
disruption or contact responses. These are possible missing signals, not verified explanations.
The observed score distribution shift warrants a shadow evaluation before any threshold is reused.

## Data sensitivity and limitations

There are 31,994 identical source rows. Retaining only one identical record
in the test evaluation gives AP 0.0267 on 19,317 rows.
This only reweights test evaluation. It does not retrain the model or prove which rows were duplicates.
Training/validation exclude 926 eligible rows whose final
labels were not available before their respective evaluation freeze.

The model remains an offline case study. Two old Portuguese hotels, missing event timestamps and
unknown group dependence limit both the evaluation and transfer to Milan. The outcome is a late
cancellation/no-show proxy, not all cancellation risk at booking creation. No intervention value,
cost savings or revenue uplift was observed.
