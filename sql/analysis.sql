-- Grain: one row per booking, including canceled bookings and no-shows.
-- Identical rows are retained because the source has no booking identifier.

DROP VIEW IF EXISTS hotel_summary;
CREATE VIEW hotel_summary AS
SELECT
    hotel,
    COUNT(*) AS bookings,
    SUM(is_canceled) AS cancellations,
    AVG(is_canceled * 1.0) AS cancellation_rate,
    SUM(CASE WHEN is_canceled = 0 THEN total_nights ELSE 0 END) AS stayed_booking_nights,
    AVG(CASE WHEN is_canceled = 0 THEN adr_for_summary END) AS mean_stayed_booking_adr
FROM bookings
GROUP BY hotel;

DROP VIEW IF EXISTS monthly_demand;
CREATE VIEW monthly_demand AS
WITH monthly AS (
    SELECT
        strftime('%Y-%m', arrival_date) AS arrival_month,
        hotel,
        COUNT(*) AS bookings,
        SUM(is_canceled) AS cancellations,
        SUM(1 - is_canceled) AS completed_bookings,
        AVG(is_canceled * 1.0) AS cancellation_rate
    FROM bookings
    GROUP BY 1, 2
)
SELECT *,
    SUM(bookings) OVER (PARTITION BY hotel ORDER BY arrival_month) AS cumulative_bookings
FROM monthly;

DROP VIEW IF EXISTS channel_summary;
CREATE VIEW channel_summary AS
SELECT
    hotel,
    market_segment,
    distribution_channel,
    COUNT(*) AS bookings,
    SUM(is_canceled) AS cancellations,
    AVG(is_canceled * 1.0) AS cancellation_rate,
    AVG(lead_time) AS mean_lead_days
FROM bookings
GROUP BY 1, 2, 3;

DROP VIEW IF EXISTS lead_time_summary;
CREATE VIEW lead_time_summary AS
SELECT hotel, lead_band, COUNT(*) AS bookings,
    SUM(is_canceled) AS cancellations,
    AVG(is_canceled * 1.0) AS cancellation_rate
FROM bookings
GROUP BY 1, 2;

DROP VIEW IF EXISTS cancellation_timing;
CREATE VIEW cancellation_timing AS
SELECT hotel,
    CASE
        WHEN days_before_arrival_canceled < 0 THEN 'After scheduled arrival'
        WHEN days_before_arrival_canceled = 0 THEN 'Arrival day'
        WHEN days_before_arrival_canceled <= 7 THEN '1-7 days before'
        WHEN days_before_arrival_canceled <= 30 THEN '8-30 days before'
        ELSE '31+ days before'
    END AS timing,
    COUNT(*) AS cancellations
FROM bookings
WHERE is_canceled = 1
GROUP BY 1, 2;
