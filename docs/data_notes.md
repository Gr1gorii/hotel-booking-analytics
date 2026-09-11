# Data notes

One row represents one booking. The source contains 40,060 resort bookings and 79,330 city bookings,
with scheduled arrival dates from 1 July 2015 to 31 August 2017. H1 is in Algarve and H2 is in Lisbon.
The publication and its supplementary data are credited to Nuno Antonio, Ana de Almeida and Luis Nunes.

[Paper and variable definitions](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/)
| [DOI](https://doi.org/10.1016/j.dib.2018.11.126)
| [Authors' supplementary files via Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6297060/supplementaryFiles)

The download reads H1.csv and H2.csv from mmc2.zip inside the supplementary archive.
Checksums in `prepare.py` pin the actual CSV contents; the wrapper ZIP can have different timestamps.
No Kaggle account or third-party dataset mirror is required.

## Cleaning choices

- Convert CamelCase names to snake_case and trim padded category strings.
- Preserve missing children; do not assume they are zero. The no-guest flag treats unknown children
  as zero only when checking for a usable guest count. The source's four missing-child rows have adults.
- Country missingness becomes `Unknown`. Empty agent and company codes become `Not recorded`.
  These codes identify source categories, not numeric magnitudes. No attempt is made to identify agencies.
- Combine `Undefined` and `SC` meal categories because the source describes both as no meal package.
- Retain identical records. There is no original booking ID to distinguish accidental duplication
  from separate bookings with identical attributes. Generated row IDs are file positions only.
- Retain zero-night and zero-guest records in descriptive booking counts. Exclude them from the model cohort.
- Flag ADR below EUR 0 or above EUR 1,000. Keep the original value and exclude only flagged values
  from `adr_for_summary`. The cutoff is a manual review assumption, not a confirmed error boundary.

## Metric definitions

Cancellation rate = canceled bookings / all bookings in the displayed selection. It includes no-shows.
The selected period always refers to scheduled arrival. Charts do not show the booking-creation curve.
Completed bookings means `is_canceled = 0`; it is not a count of guests or occupied rooms.

Booking nights = weekend nights + weekday nights for a booking. The SQL's stayed booking nights
sum this value only for completed bookings. They are not validated room nights because room quantity
is unavailable. Mean stayed booking ADR is a booking-level average after the explicit ADR flag;
it is not a room-night-weighted hotel ADR.

Market segment and distribution channel are different dimensions. TA means travel agent and TO
means tour operator. `Direct` in one field does not guarantee identical membership in the other.
The `deposit_type` field is reconstructed from payments, so its name should not be interpreted as
a clean experimental treatment or simply the rate plan selected at booking.

## What cannot be estimated

No occupancy rate or RevPAR: the room inventory denominator is missing. No verified lost revenue:
payments, refunds, resale, commissions and room quantities are not sufficiently represented.
No causal channel effect: prices, lead times, customers and contract terms differ between groups.
No annual growth comparison using full totals: 2015 and 2017 are partial years.
No automatic extrapolation to Milan: two historical Portuguese hotels are not a current market sample.

The source reports timestamps reconstructed relative to D-1, but does not expose the full change
history. Earlier forecasts would require new data. Similar bookings, repeat guests and hotel groups
can be dependent, while anonymous identifiers prevent a complete dependence audit.
