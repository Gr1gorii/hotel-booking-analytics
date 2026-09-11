# Cancellation model

## Decision and prediction time

The intended decision is which remaining bookings deserve a manual arrival confirmation.
The research cutoff is the end of the day before scheduled arrival (D-1), using hotel-local dates.
The source has dates, not timestamps or a documented timezone. A true production cutoff cannot be verified.

The authors describe extracting booking values from change logs relative to the day before arrival.
They do not provide a booking-creation snapshot. Training on every row would include many bookings
already canceled before the prediction point. That would answer a retrospective classification
question rather than the intended operational question.

## Population and target

- Include bookings entered at least two days before arrival, with an overnight stay and at least one guest.
- Remove cancellations recorded before D-1 because their outcome was already known.
- Remove cancellations recorded on D-1 because the within-day ordering is unknown.
- Among the remaining bookings, predict `is_canceled`, including no-shows.
- Remove inconsistent completed-booking dates if present.

This narrows the question to late cancellations and no-shows. The last status date is a proxy for
event timing, used only to define eligibility and label availability. Delayed administrative updates
could misclassify the risk set. Transaction and event logs are needed to verify it.

## Feature choices

Fourteen inputs describe the booked stay, calendar, hotel and channel. Their use depends on the
paper's documented D-1 reconstruction. The full field review is in `docs/feature_audit.csv`.
All preprocessing is fitted on training rows only. Missing children are imputed inside the pipeline;
unseen categories are ignored by the encoder.

Status, status date, assigned room, deposit, ADR, booking changes and historical outcome counts
are not inputs. Some have direct outcome information; others have unclear update timing.
Omitting an input is not a claim that it always leaks in every PMS.

## Evaluation

Split on prediction date, not randomly. The first freeze is 1 July 2016 and the final freeze is
1 January 2017. Training and validation rows must also have a final status date strictly before
their respective freeze. This prevents learning from an unresolved stay at the boundary.
Exact included date ranges and exclusions are saved in `artifacts/model_metadata.json`.

Compare training prevalence, a fixed `lead_time >= 90` rule, logistic regression and small gradient
boosting. Hyperparameters are fixed. Choose between the learned models using validation average
precision. Do not refit after selection. Evaluate the frozen models on test predictions from
1 January to 30 August 2017 (arrivals from 2 January to 31 August).

Each learned model's threshold is its validation 90th score percentile. Ties and time drift can
change the workload. The rule flags all bookings with lead time of at least 90 days. Precision and
recall must be read together with review rate because the methods do not have equal workloads.
The prevalence baseline flags no bookings at 0.5. The binary rule is not a probability forecast;
its Brier score is left blank.

Average precision summarizes the precision-recall ranking and has prevalence as its no-skill
reference. It is not accuracy. Brier score measures probability error. Calibration bins and errors
by hotel, segment, lead time and month help explain the limits. Undefined slice precision or recall
is left missing when its denominator is zero.

The 95% AP intervals resample arrival dates in 200 bootstrap draws. They allow some same-day
dependence, but not all group or repeat-guest dependence. The intervals do not validate transfer
to a new hotel. Permutation importance is a post-evaluation diagnostic, not a feature-selection step.

## Current use

Keep this model offline. The observed lift is small, probabilities are imperfect and the validation
review threshold does not preserve its workload on test data. No business impact has been measured.
Never convert a risk score directly into a booking rejection, a deposit policy or an overbooking level.

For a Milan pilot, collect timestamped booking snapshots, cancellation events, group identifiers,
actual room inventory, prices at prediction time and intervention outcomes. Start with a shadow
evaluation on that hotel's data. Measure useful confirmations, contact cost and false alarms before
testing any intervention. This dataset supports neither current Milan demand estimates nor causal
claims about changing booking terms.
