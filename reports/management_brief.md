# Five findings for a hotel manager

Scope: all 119,390 source bookings, July 2015 to August 2017. Identical records are retained.
These are observations from two Portuguese hotels, followed by proposed actions to test.

1. **The city hotel had more cancellations.** Its cancellation rate was 41.7%,
   compared with 27.8% at the resort. Use separate hotel baselines in planning.
   Do not interpret the difference as an effect of hotel type; there is only one hotel of each type.

2. **October led the complete-year arrival counts.** In 2016, the only complete year, the city hotel's largest
   scheduled-arrival month was October; the resort's was also
   October. Build separate monthly planning views and compare
   gross bookings with completed stays. Booking counts alone do not establish staffing needs.

3. **Channel mix deserves attention.** Online TA bookings had a 36.7%
   cancellation rate, direct bookings 15.3%, and group bookings 61.1%.
   Review group release dates and channel terms, then compare similar lead-time and hotel cohorts.
   Do not shift budget based only on these unadjusted rates; commissions and acquisition costs are absent.

4. **Long-lead bookings carry more cancellation uncertainty overall.** The rate was
   57.0% beyond 180 days, versus 9.6% at 0-7 days.
   Track booking pickup and cancellations by lead-time band. This pattern does not automatically
   survive in the D-1 model cohort, where most earlier cancellations have already been removed.

5. **Most cancellations are known before the last week.** The median recorded cancellation was
   54 days before arrival; 85.7% were recorded more than seven days before.
   An earlier confirmation workflow may be more useful than a last-day score. Test that idea with
   timestamped booking snapshots, contact outcomes and costs. The current table cannot measure its effect.

The data do not establish occupancy, RevPAR, recoverable revenue or the effect of any intervention.
Revalidate the definitions and all recommendations with a Milan hotel's current PMS data.

Source: [Antonio, de Almeida and Nunes (2019)](https://doi.org/10.1016/j.dib.2018.11.126).
Calculations: `sql/analysis.sql` and `src/hotel_booking/report.py`.
