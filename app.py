"""Run with: streamlit run app.py"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from hotel_booking.prepare import ROOT  # noqa: E402

BOOKINGS = ROOT / "data/processed/bookings.parquet"
if not BOOKINGS.exists():
    BOOKINGS = ROOT / "demo/bookings.parquet"
PREDICTIONS = ROOT / "artifacts/test_predictions.parquet"
if not PREDICTIONS.exists():
    PREDICTIONS = ROOT / "demo/test_predictions.parquet"

st.set_page_config(page_title="Hotel Booking Intelligence", page_icon="🏨", layout="wide")
st.html("""
<style>
    .stMainBlockContainer {max-width: 1480px; padding-top: 3rem;}
    h1 {letter-spacing: -0.035em;}
    h3 {font-size: 1.2rem !important;}
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid #e3e9ee; border-top: 3px solid #087e8b;
        border-radius: 8px; padding: 16px 18px;
    }
    [data-testid="stMetricValue"] {font-size: clamp(1.3rem, 2.8vw, 2.1rem);}
    [data-testid="stPlotlyChart"] {background: #fff; border-radius: 8px; padding: 8px;}
    [data-baseweb="tab-list"] {gap: 20px; margin-bottom: 16px;}
    @media (max-width: 640px) {
        .stMainBlockContainer {padding-top: 2rem;}
        [data-testid="stMetric"] {padding: 10px 12px;}
        [data-testid="stMetricValue"] {font-size: 1.6rem;}
    }
</style>
""")
COLORS = {"City Hotel": "#087E8B", "Resort Hotel": "#C17B35"}
LEAD_ORDER = ["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181+ days"]


@st.cache_data
def load_data():
    return pd.read_parquet(BOOKINGS)


def chart(fig, percent=False):
    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=13, color="#142C3B"),
        margin=dict(l=10, r=20, t=35, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_yaxes(gridcolor="#E4E8ED", rangemode="tozero")
    if percent:
        fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def grouped_rates(df, dimensions):
    result = (
        df.groupby(dimensions, observed=True)
        .agg(
            bookings=("row_id", "size"),
            cancellations=("is_canceled", "sum"),
        )
        .reset_index()
    )
    result["cancellation_rate"] = result.cancellations / result.bookings
    # Wilson intervals describe sampling uncertainty, not causal effects.
    n, p, z = result.bookings, result.cancellation_rate, 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    result["error_plus"] = center + radius - p
    result["error_minus"] = p - center + radius
    return result


st.title("Hotel Booking Intelligence")
st.caption(
    "Booking demand, channel differences and late cancellation risk | Portugal, July 2015 to August 2017"
)

if not BOOKINGS.exists():
    st.info("The dataset has not been prepared yet. Follow the two setup commands in the README.")
    st.code("python run.py prepare\npython run.py train", language="bash")
    st.stop()

bookings = load_data()
quality = json.loads((ROOT / "artifacts/data_quality.json").read_text())
with st.sidebar:
    st.header("Explore bookings")
    st.caption(
        "These filters apply to Overview, Channels and the data export. Model evaluation uses its fixed test set."
    )
    hotel = st.selectbox("Hotel", ["All hotels", "City Hotel", "Resort Hotel"], key="hotel_filter")
    period = st.selectbox(
        "Arrival period",
        ["Full dataset", "2016 (complete year)", "2015 (Jul-Dec)", "2017 (Jan-Aug)"],
        key="period_filter",
    )
    segments = st.multiselect(
        "Market segment", sorted(bookings.market_segment.unique()), key="segment_filter"
    )
    if st.button("Reset filters"):
        st.session_state.pop("hotel_filter", None)
        st.session_state.pop("period_filter", None)
        st.session_state.pop("segment_filter", None)
        st.rerun()
    st.divider()
    st.markdown("[Original publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/)")
    st.caption(
        "One city hotel in Lisbon and one resort in Algarve. This is a historical case study, not a forecast for Milan."
    )

filtered = bookings.copy()
if hotel != "All hotels":
    filtered = filtered[filtered.hotel == hotel]
if period != "Full dataset":
    filtered = filtered[filtered.arrival_date.dt.year == int(period[:4])]
if segments:
    filtered = filtered[filtered.market_segment.isin(segments)]

overview, channels, model_tab, data_tab = st.tabs(
    ["Overview", "Channels & decisions", "Cancellation model", "Data & SQL"]
)

with overview:
    if filtered.empty:
        st.info("No bookings match these filters. Choose another period or reset the filters.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Bookings", f"{len(filtered):,}")
        b.metric("Cancellation rate", f"{filtered.is_canceled.mean():.1%}")
        c.metric("Completed bookings", f"{int((1 - filtered.is_canceled).sum()):,}")
        d.metric("Median lead time", f"{filtered.lead_time.median():.0f} days")
        st.caption(
            "Cancellation includes no-shows. The denominator is every booking in the selected view, including zero-night records."
        )
        trend = filtered.assign(month=filtered.arrival_date.dt.to_period("M").dt.to_timestamp())
        trend = grouped_rates(trend, ["month", "hotel"])
        left, right = st.columns(2)
        with left:
            st.subheader("Scheduled arrivals by month")
            chart(
                px.line(
                    trend,
                    x="month",
                    y="bookings",
                    color="hotel",
                    markers=True,
                    color_discrete_map=COLORS,
                    labels={"month": "", "bookings": "Bookings"},
                )
            )
        with right:
            st.subheader("Cancellation rate by arrival month")
            chart(
                px.line(
                    trend,
                    x="month",
                    y="cancellation_rate",
                    color="hotel",
                    markers=True,
                    color_discrete_map=COLORS,
                    labels={"month": "", "cancellation_rate": "Cancellation rate"},
                    hover_data={"cancellation_rate": ":.1%", "bookings": True},
                ),
                percent=True,
            )
        st.caption(
            "The timeline is grouped by scheduled arrival, not booking creation or cancellation date. Annual totals are not comparable: only 2016 is complete."
        )
        st.subheader("Hotel comparison")
        by_hotel = grouped_rates(filtered, ["hotel"])
        chart(
            px.bar(
                by_hotel,
                x="hotel",
                y="cancellation_rate",
                error_y="error_plus",
                error_y_minus="error_minus",
                color="hotel",
                color_discrete_map=COLORS,
                hover_data={
                    "cancellation_rate": ":.1%",
                    "bookings": True,
                    "cancellations": True,
                    "error_plus": False,
                    "error_minus": False,
                },
                labels={"hotel": "", "cancellation_rate": "Cancellation rate"},
            ),
            percent=True,
        )
        st.caption(
            "Bars show 95% Wilson intervals. Group bookings may be dependent, so these intervals can be too narrow. Hotel mix and booking terms can explain part of the difference."
        )

with channels:
    if filtered.empty:
        st.info("No bookings match these filters.")
    else:
        st.subheader("Market segment and booking lead time")
        left, right = st.columns(2)
        with left:
            rates = grouped_rates(filtered, ["market_segment", "hotel"])
            segment_chart = px.bar(
                rates,
                x="cancellation_rate",
                y="market_segment",
                color="hotel",
                barmode="group",
                orientation="h",
                color_discrete_map=COLORS,
                labels={"market_segment": "", "cancellation_rate": "Cancellation rate"},
                hover_data={"cancellation_rate": ":.1%", "bookings": True},
            )
            segment_chart.update_xaxes(tickformat=".0%", rangemode="tozero")
            chart(segment_chart)
        with right:
            rates = grouped_rates(filtered, ["lead_band", "hotel"])
            rates["lead_band"] = pd.Categorical(rates.lead_band, categories=LEAD_ORDER, ordered=True)
            rates = rates.sort_values(["hotel", "lead_band"])
            chart(
                px.line(
                    rates,
                    x="lead_band",
                    y="cancellation_rate",
                    color="hotel",
                    markers=True,
                    color_discrete_map=COLORS,
                    category_orders={"lead_band": LEAD_ORDER},
                    labels={"lead_band": "", "cancellation_rate": "Cancellation rate"},
                    hover_data={"cancellation_rate": ":.1%", "bookings": True},
                ),
                percent=True,
            )
        st.caption(
            "Market segment and distribution channel are different fields. Small segments are unstable; inspect the booking counts before comparing rates."
        )
        table = grouped_rates(filtered, ["distribution_channel", "hotel"])
        st.dataframe(
            table[["distribution_channel", "hotel", "bookings", "cancellations", "cancellation_rate"]],
            hide_index=True,
            width="stretch",
            column_config={"cancellation_rate": st.column_config.NumberColumn(format="percent")},
        )
        st.subheader("When cancellations were recorded")
        canceled = filtered[filtered.is_canceled == 1].copy()
        if canceled.empty:
            st.info("No cancellations in this selection.")
        else:
            canceled["timing"] = pd.cut(
                canceled.days_before_arrival_canceled,
                [-np.inf, -1, 0, 7, 30, np.inf],
                labels=[
                    "After arrival",
                    "Arrival day",
                    "1-7 days before",
                    "8-30 days before",
                    "31+ days before",
                ],
            )
            timing = (
                canceled.groupby(["timing", "hotel"], observed=True).size().reset_index(name="cancellations")
            )
            chart(
                px.bar(
                    timing,
                    x="timing",
                    y="cancellations",
                    color="hotel",
                    color_discrete_map=COLORS,
                    labels={"timing": "", "cancellations": "Canceled bookings"},
                )
            )
        st.caption(
            "Timing uses the last reservation status date. It is an administrative date, not a timestamped cancellation event log."
        )
    with st.expander("Five findings and proposed management actions (full dataset)"):
        path = ROOT / "reports/management_brief.md"
        if path.exists():
            st.markdown(path.read_text())

with model_tab:
    if not (ROOT / "artifacts/model_metadata.json").exists():
        st.info("Run python run.py train to create the evaluation.")
    else:
        meta = json.loads((ROOT / "artifacts/model_metadata.json").read_text())
        metrics = pd.read_csv(ROOT / "artifacts/model_metrics.csv")
        predictions = pd.read_parquet(PREDICTIONS)
        st.subheader("Can we identify the remaining late cancellations?")
        st.markdown(
            "The prediction point is the end of the day before arrival. Bookings already canceled, cancellations recorded on that day and bookings created too late are excluded. The target includes arrival-day cancellations and no-shows."
        )
        st.info(
            "Research result: the signal is weak. Keep the model in offline evaluation; the results do not justify automated booking decisions."
        )
        a, b, c = st.columns(3)
        a.metric("Test bookings", f"{len(predictions):,}")
        b.metric("Late cancellation rate", f"{predictions.is_canceled.mean():.2%}")
        c.metric("Selected on validation", meta["selected_model"])
        st.caption(
            "Fixed test arrivals: 2 January to 31 August 2017. Sidebar filters do not change this evaluation."
        )
        view = metrics[
            ["model", "average_precision", "roc_auc", "brier_score", "precision", "recall", "review_rate"]
        ].copy()
        for column in ["average_precision", "roc_auc", "brier_score"]:
            view[column] = view[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "n/a")
        for column in ["precision", "recall", "review_rate"]:
            view[column] = view[column].map(lambda value: f"{value:.1%}")
        view.columns = ["Method", "Avg. precision", "ROC AUC", "Brier", "Precision", "Recall", "Review share"]
        st.dataframe(view, hide_index=True, width="stretch")
        st.caption(
            "Average precision and ROC AUC measure ranking. Brier score measures probability error (lower is better). The binary rule has no probability forecast, so Brier is not applicable. Each learned model uses its own threshold chosen to flag about 10% of validation bookings; test workloads differ."
        )
        with st.expander("Feature availability and evaluation protocol"):
            st.markdown((ROOT / "docs/model_card.md").read_text())
            st.dataframe(pd.read_csv(ROOT / "docs/feature_audit.csv"), hide_index=True, width="stretch")
            st.dataframe(pd.DataFrame(meta["splits"]), hide_index=True, width="stretch")
        left, right = st.columns(2)
        with left:
            st.subheader("Uncertainty in average precision")
            fig = go.Figure(
                go.Scatter(
                    x=metrics.average_precision,
                    y=metrics.model,
                    mode="markers",
                    marker=dict(color="#087E8B", size=10),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=metrics.ap_upper - metrics.average_precision,
                        arrayminus=metrics.average_precision - metrics.ap_lower,
                    ),
                )
            )
            fig.update_xaxes(title="Average precision", rangemode="tozero")
            chart(fig)
            st.caption(
                "95% percentile intervals from 200 resamples of arrival dates. These do not measure uncertainty about transfer to another hotel."
            )
        with right:
            st.subheader("Do probabilities match outcomes?")
            calibration = pd.read_csv(ROOT / "artifacts/calibration.csv")
            limit = max(calibration.mean_prediction.max(), calibration.observed_rate.max()) * 1.1
            fig = px.scatter(
                calibration,
                x="mean_prediction",
                y="observed_rate",
                size="rows",
                color_discrete_sequence=["#087E8B"],
                hover_data={"rows": True},
                labels={
                    "mean_prediction": "Mean predicted probability",
                    "observed_rate": "Observed cancellation rate",
                },
            )
            fig.add_shape(
                type="line", x0=0, y0=0, x1=limit, y1=limit, line=dict(color="#7D8992", dash="dash")
            )
            fig.update_xaxes(range=[0, limit], tickformat=".0%")
            fig.update_yaxes(range=[0, limit])
            chart(fig, percent=True)
            st.caption(
                "Each point is a group of similar test scores. The diagonal represents perfect calibration."
            )
        st.subheader("Review workload explorer")
        st.caption(
            "This is a retrospective threshold scenario. It does not change the locked benchmark or estimate cancellations prevented."
        )
        threshold = st.slider("Review threshold (%)", 0.0, 20.0, float(meta["threshold"] * 100), 0.1) / 100
        flagged = predictions.probability >= threshold
        tp = int((flagged & predictions.is_canceled.eq(1)).sum())
        fp = int((flagged & predictions.is_canceled.eq(0)).sum())
        fn = int((~flagged & predictions.is_canceled.eq(1)).sum())
        a, b, c, d = st.columns(4)
        a.metric("Bookings to review", f"{int(flagged.sum()):,}")
        b.metric("Cancellations flagged", f"{tp:,}")
        c.metric("False alarms", f"{fp:,}")
        d.metric("Cancellations missed", f"{fn:,}")
        st.subheader("Where the frozen model makes mistakes")
        slices = pd.read_csv(ROOT / "artifacts/error_slices.csv")
        dimension = st.selectbox(
            "Break down errors by", ["hotel", "market_segment", "lead_band", "arrival_month"]
        )
        st.dataframe(
            slices[slices.dimension == dimension].drop(columns="dimension"), hide_index=True, width="stretch"
        )
        with st.expander("Inspect ten wrong predictions"):
            st.dataframe(pd.read_csv(ROOT / "artifacts/error_examples.csv"), hide_index=True, width="stretch")
            st.caption(
                "Examples are selected by model confidence, not representative sampling. Row IDs identify source rows, not guests."
            )
        with st.expander("Which features carry signal?"):
            importance = pd.read_csv(ROOT / "artifacts/feature_importance.csv").sort_values("ap_drop")
            chart(
                px.bar(
                    importance,
                    x="ap_drop",
                    y="feature",
                    orientation="h",
                    color_discrete_sequence=["#087E8B"],
                    labels={"ap_drop": "Average precision drop after shuffling", "feature": ""},
                )
            )
            st.caption(
                "Post-evaluation diagnostic, three shuffles per feature. Correlated inputs can share importance. No causal interpretation."
            )

with data_tab:
    st.subheader("What is in the source?")
    st.markdown(
        "119,390 bookings from two Portuguese hotels. H1 and H2 come from the authors' supplementary archive. Downloaded files are verified with SHA-256 checksums."
    )
    a, b, c = st.columns(3)
    a.metric("Identical rows retained", f"{quality['duplicate_rows']:,}")
    b.metric("Zero-night bookings", f"{quality['zero_nights']:,}")
    c.metric("ADR values flagged", quality["adr_flagged"])
    st.markdown(
        "Identical rows may describe separate group bookings. Missing children remain missing; anonymous agent and company codes are treated as categories. Two ADR values outside EUR 0-1,000 are excluded only from the ADR summary. This is an explicit review rule, not a verified correction."
    )
    st.subheader("From the full dataset to the model cohort")
    st.dataframe(
        pd.DataFrame(quality["cohort_waterfall"].items(), columns=["Reason", "Bookings"]), hide_index=True
    )
    st.caption(
        "Exclusion reasons are mutually exclusive. No-stay and late-creation rules take precedence over cancellation timing."
    )
    st.subheader("Preview and export the selected bookings")
    export_columns = [
        "row_id",
        "hotel",
        "arrival_date",
        "market_segment",
        "distribution_channel",
        "lead_time",
        "total_nights",
        "is_canceled",
        "adr",
        "adr_flag",
        "duplicate_record",
    ]
    st.caption(
        f"Preview: first {min(100, len(filtered)):,} of {len(filtered):,} rows in the current sidebar selection. The CSV contains every selected row."
    )
    st.dataframe(filtered[export_columns].head(100), hide_index=True, width="stretch")
    st.download_button(
        "Download selected bookings",
        filtered[export_columns].to_csv(index=False),
        file_name="selected_bookings.csv",
        mime="text/csv",
    )
    with st.expander("SQL used for the saved analytical tables"):
        st.code((ROOT / "sql/analysis.sql").read_text(), language="sql")
    with st.expander("Definitions and limits"):
        st.markdown((ROOT / "docs/data_notes.md").read_text())

st.divider()
st.caption(
    "Source: Antonio, de Almeida and Nunes (2019), Hotel booking demand datasets. DOI: 10.1016/j.dib.2018.11.126. Data and publication: CC BY 4.0."
)
