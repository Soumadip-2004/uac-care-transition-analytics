"""
Care Transition Efficiency & Placement Outcome Analytics
Streamlit dashboard

Run with:  streamlit run app.py
Expects uac_pipeline_features.csv (produced by the analysis notebook)
in the same folder as this file.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Care Transition Efficiency & Placement Outcome Analytics",
    layout="wide",
)

# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------
@st.cache_data
def load_data(path="uac_pipeline_features.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    return df

df = load_data()

st.title("Care Transition Efficiency & Placement Outcome Analytics")
st.caption("UAC Program — care pipeline flow, transition efficiency, and placement outcome monitoring")

# ---------------------------------------------------------
# Sidebar: user controls
# ---------------------------------------------------------
st.sidebar.header("Filters")

min_date, max_date = df["date"].min().date(), df["date"].max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
fdf = df.loc[mask].copy()

st.sidebar.divider()
st.sidebar.subheader("Metric toggles")
show_transfer_ratio = st.sidebar.checkbox("Transfer Efficiency Ratio", value=True)
show_discharge_eff = st.sidebar.checkbox("Discharge Effectiveness", value=True)
show_throughput = st.sidebar.checkbox("Pipeline Throughput Rate", value=True)

st.sidebar.divider()
st.sidebar.subheader("Alert thresholds")
backlog_alert_days = st.sidebar.slider(
    "Backlog alert: sustained imbalance (days)", min_value=3, max_value=30, value=7
)
drop_alert_pct = st.sidebar.slider(
    "Discharge sudden-drop alert (% below 7-day avg)", min_value=10, max_value=80, value=40
)

if fdf.empty:
    st.warning("No data in the selected date range.")
    st.stop()

# ---------------------------------------------------------
# KPI row
# ---------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Avg Transfer Efficiency Ratio", f"{fdf['transfer_efficiency_ratio'].mean():.2f}")
k2.metric("Avg Discharge Effectiveness", f"{fdf['discharge_effectiveness'].mean():.2f}")
latest_throughput = fdf["throughput_rate_30d"].dropna()
k3.metric("Latest 30d Throughput Rate", f"{latest_throughput.iloc[-1]:.2f}" if len(latest_throughput) else "N/A")
k4.metric("Current Backlog Level", f"{fdf['backlog_accum'].iloc[-1]:,.0f}")
k5.metric("Avg Outcome Stability (CV)", f"{fdf['discharge_cv_30d'].mean():.2f}")

st.divider()

# ---------------------------------------------------------
# Module 1: Care Pipeline Flow Visualization
# ---------------------------------------------------------
st.subheader("Care Pipeline Flow")

col1, col2 = st.columns([2, 1])

with col1:
    fig = go.Figure()
    stage_cols = ["apprehended", "cbp_custody", "transferred_out_cbp", "hhs_care", "discharged"]
    stage_labels = ["Apprehended (CBP intake)", "CBP Custody", "Transferred out of CBP",
                     "HHS Care", "Discharged"]
    for col, label in zip(stage_cols, stage_labels):
        fig.add_trace(go.Scatter(x=fdf["date"], y=fdf[col], name=label, mode="lines"))
    fig.update_layout(height=420, xaxis_title="Date", yaxis_title="Children",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, width='stretch')

with col2:
    # Snapshot flow diagram (average volumes over selected range)
    avg_vals = fdf[stage_cols].mean()
    fig_flow = go.Figure(go.Funnel(
        y=stage_labels,
        x=avg_vals.values,
        textinfo="value",
    ))
    fig_flow.update_layout(height=420, title="Avg Stage Volume (selected range)")
    st.plotly_chart(fig_flow, width='stretch')

st.divider()

# ---------------------------------------------------------
# Module 2: Transfer & Discharge Efficiency Panels
# ---------------------------------------------------------
st.subheader("Transfer & Discharge Efficiency")

eff_fig = go.Figure()
if show_transfer_ratio:
    eff_fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["transfer_efficiency_ratio"],
                                  name="Transfer Efficiency Ratio", mode="lines"))
if show_discharge_eff:
    eff_fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["discharge_effectiveness"],
                                  name="Discharge Effectiveness", mode="lines"))
if show_throughput:
    eff_fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["throughput_rate_30d"],
                                  name="30d Throughput Rate", mode="lines"))

eff_fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
eff_fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Ratio",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(eff_fig, width='stretch')

st.caption("Dashed line = ratio of 1.0 (inflow/outflow parity). Toggle series from the sidebar.")

st.divider()

# ---------------------------------------------------------
# Module 3: Bottleneck Detection Charts
# ---------------------------------------------------------
st.subheader("Bottleneck & Backlog Detection")

bcol1, bcol2 = st.columns(2)

with bcol1:
    backlog_fig = go.Figure()
    backlog_fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["backlog_accum"],
                                      name="Cumulative backlog signal", fill="tozeroy"))
    backlog_fig.add_hline(y=0, line_dash="dash", line_color="gray")
    backlog_fig.update_layout(height=380, xaxis_title="Date", yaxis_title="Net accumulated children")
    st.plotly_chart(backlog_fig, width='stretch')

with bcol2:
    # Recompute sustained-imbalance streaks against the user's chosen threshold
    tmp = fdf.copy()
    tmp["imbalance"] = tmp["net_hhs_flow"] > 0
    tmp["streak_id"] = (tmp["imbalance"] != tmp["imbalance"].shift()).cumsum()
    streaks = (
        tmp[tmp["imbalance"]]
        .groupby("streak_id")
        .agg(start=("date", "min"), end=("date", "max"), days=("date", "count"))
    )
    flagged = streaks[streaks["days"] >= backlog_alert_days].sort_values("days", ascending=False)

    st.markdown(f"**Sustained backlog periods (≥ {backlog_alert_days} days)**")
    if flagged.empty:
        st.success("No sustained backlog periods at this threshold.")
    else:
        st.dataframe(
            flagged.rename(columns={"start": "Start", "end": "End", "days": "Days"}),
            width='stretch',
        )
        st.warning(f"{len(flagged)} backlog period(s) flagged in the selected range.")

st.divider()

# ---------------------------------------------------------
# Module 4: Outcome Trend Analysis
# ---------------------------------------------------------
st.subheader("Outcome Trend Analysis")

ocol1, ocol2 = st.columns(2)

with ocol1:
    stab_fig = go.Figure()
    stab_fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["discharged"], name="Daily discharges",
                                   mode="lines", line=dict(color="lightgray")))
    stab_fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["discharge_roll_mean_30d"],
                                   name="30d rolling mean", mode="lines"))
    stab_fig.update_layout(height=380, xaxis_title="Date", yaxis_title="Children discharged")
    st.plotly_chart(stab_fig, width='stretch')

with ocol2:
    # Sudden-drop alerts against user-selected threshold
    tmp2 = fdf.copy()
    tmp2["drop_alert"] = tmp2["pct_drop_vs_7d"] < -(drop_alert_pct / 100)
    drops = tmp2[tmp2["drop_alert"]][["date", "discharged", "discharge_7d_avg", "pct_drop_vs_7d"]]

    st.markdown(f"**Sudden discharge drops (> {drop_alert_pct}% below 7-day avg)**")
    if drops.empty:
        st.success("No sudden drops flagged at this threshold.")
    else:
        drops = drops.rename(columns={
            "date": "Date", "discharged": "Discharged",
            "discharge_7d_avg": "7d Avg", "pct_drop_vs_7d": "% Drop",
        })
        drops["% Drop"] = (drops["% Drop"] * 100).round(1)
        st.dataframe(drops, width='stretch')
        st.warning(f"{len(drops)} sudden-drop day(s) flagged in the selected range.")

st.divider()

# ---------------------------------------------------------
# Weekday / monthly pattern (supporting view)
# ---------------------------------------------------------
st.subheader("Temporal Patterns")

tcol1, tcol2 = st.columns(2)

with tcol1:
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_avg = fdf.groupby("weekday")[["transferred_out_cbp", "discharged"]].mean().reindex(weekday_order)
    fig_wd = px.bar(weekday_avg, barmode="group",
                     labels={"value": "Avg children", "weekday": "Day of week"})
    fig_wd.update_layout(height=350, title="Avg Transfers & Discharges by Weekday")
    st.plotly_chart(fig_wd, width='stretch')

with tcol2:
    monthly = fdf.groupby("year_month")[["apprehended", "transferred_out_cbp", "discharged"]].sum()
    fig_month = px.line(monthly, labels={"value": "Total children", "year_month": "Month"})
    fig_month.update_layout(height=350, title="Month-over-Month Volumes")
    st.plotly_chart(fig_month, width='stretch')

st.divider()
st.caption(
    "Data source: HHS Unaccompanied Alien Children Program. "
    "Metrics engineered in the companion analysis notebook."
)