import re
from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Federated GRU Diagnostics", page_icon="📈", layout="wide")
st.title("📈 Federated GRU Diagnostics Dashboard")
st.caption("Standalone vs Federated GRU · Sensor Forecasting")

PROJECT_DIR = Path(__file__).resolve().parent
COMPARISON_PATH = PROJECT_DIR / "gru_model_comparison.csv"
STANDALONE_DIR = PROJECT_DIR.parent / "gru_vibration_results"

SENSOR_NAMES = {"vibration_x": "Vibration X", "vibration_y": "Vibration Y", "vibration_z": "Vibration Z"}

@st.cache_data
def load_comparison(path, uploaded_bytes=None):
    return pd.read_csv(BytesIO(uploaded_bytes)) if uploaded_bytes is not None else pd.read_csv(path)

@st.cache_data
def load_predictions(path):
    return pd.read_csv(path)

st.sidebar.header("Controls")
uploaded_comparison = st.sidebar.file_uploader("Upload comparison CSV (optional)", type=["csv"])

if uploaded_comparison:
    comparison = load_comparison(COMPARISON_PATH, uploaded_comparison.getvalue())
elif COMPARISON_PATH.exists():
    comparison = load_comparison(COMPARISON_PATH)
else:
    st.error(f"Comparison CSV not found: {COMPARISON_PATH}")
    st.info("Run `python compare_gru_models.py` first, or upload gru_model_comparison.csv.")
    st.stop()

required = {"site", "sensor", "standalone_rmse", "federated_rmse", "standalone_mae", "federated_mae",
            "standalone_r2", "federated_r2", "rmse_improvement_percent", "mae_improvement_percent"}
missing = required - set(comparison.columns)
if missing:
    st.error(f"Missing columns: {', '.join(sorted(missing))}")
    st.stop()

comparison["sensor_label"] = comparison["sensor"].map(SENSOR_NAMES).fillna(comparison["sensor"])
sites = sorted(comparison["site"].unique())
sensors = sorted(comparison["sensor"].unique())

site_filter = st.sidebar.selectbox("Site", ["All sites"] + sites)
sensor_filter = st.sidebar.selectbox("Sensor", ["All sensors"] + [SENSOR_NAMES.get(s, s) for s in sensors])

filtered = comparison.copy()
if site_filter != "All sites":
    filtered = filtered[filtered.site == site_filter]
if sensor_filter != "All sensors":
    key = next(k for k, v in SENSOR_NAMES.items() if v == sensor_filter)
    filtered = filtered[filtered.sensor == key]

# Note: script's improvement formula is (standalone error - FL error)/standalone error.
# Positive = FL error lower; negative = FL error higher.
avg_rmse = filtered["rmse_improvement_percent"].mean()
avg_mae = filtered["mae_improvement_percent"].mean()
fl_better = (filtered["rmse_improvement_percent"] > 0).sum()

st.subheader("Summary")
a, b, c, d = st.columns(4)
a.metric("Comparisons", len(filtered))
b.metric("Mean RMSE improvement (FL)", f"{avg_rmse:+.1f}%")
c.metric("Mean MAE improvement (FL)", f"{avg_mae:+.1f}%")
d.metric("FL better by RMSE", f"{fl_better}/{len(filtered)}")
st.caption("Positive improvement means FL has lower error; negative means standalone has lower error.")

def paired_bar(df, standalone_col, federated_col, metric_name):
    long = df.melt(id_vars=["site", "sensor_label"], value_vars=[standalone_col, federated_col],
                   var_name="model", value_name=metric_name)
    long["model"] = long["model"].map({standalone_col: "Standalone GRU", federated_col: "Federated GRU"})
    fig = px.bar(long, x="sensor_label", y=metric_name, color="model", barmode="group",
                 facet_col="site" if site_filter == "All sites" else None,
                 title=f"{metric_name} by site and sensor")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
paired_bar(filtered, "standalone_rmse", "federated_rmse", "RMSE")
paired_bar(filtered, "standalone_mae", "federated_mae", "MAE")

st.subheader("RMSE improvement (%)")
change = px.bar(filtered, x="sensor_label", y="rmse_improvement_percent",
                color="rmse_improvement_percent",
                facet_col="site" if site_filter == "All sites" else None,
                title="Positive favors federated; negative favors standalone")
change.add_hline(y=0, line_dash="dash")
st.plotly_chart(change, use_container_width=True)

st.subheader("R² comparison")
r2 = filtered.melt(id_vars=["site", "sensor_label"], value_vars=["standalone_r2", "federated_r2"],
                   var_name="model", value_name="R²")
r2["model"] = r2["model"].map({"standalone_r2": "Standalone GRU", "federated_r2": "Federated GRU"})
r2fig = px.bar(r2, x="sensor_label", y="R²", color="model", barmode="group",
                facet_col="site" if site_filter == "All sites" else None)
r2fig.add_hline(y=0, line_dash="dash")
st.plotly_chart(r2fig, use_container_width=True)

st.subheader("Comparison table")
cols = ["site", "sensor", "test_windows", "test_points", "standalone_rmse", "federated_rmse",
        "rmse_improvement_percent", "standalone_mae", "federated_mae", "mae_improvement_percent",
        "standalone_r2", "federated_r2"]
cols = [c for c in cols if c in filtered.columns]
st.dataframe(filtered[cols].round(6), use_container_width=True, hide_index=True)
st.download_button("Download filtered results", filtered[cols].to_csv(index=False),
                   file_name="filtered_gru_comparison.csv", mime="text/csv")

st.divider()
st.subheader("Standalone prediction explorer")
st.caption("Displays saved standalone predictions. FL point-level predictions are not yet saved.")
pred_site = st.selectbox("Site for prediction view", sites)
pred_sensor = st.selectbox("Sensor for prediction view", sensors, format_func=lambda x: SENSOR_NAMES.get(x, x))
pred_path = STANDALONE_DIR / f"{pred_site}_{pred_sensor}_predictions.csv"

if pred_path.exists():
    pred = load_predictions(pred_path)
    if {"window_id", "forecast_step", "actual", "predicted"}.issubset(pred.columns):
        ids = sorted(pred.window_id.unique())
        window = st.slider("Forecast window", int(min(ids)), int(max(ids)), int(min(ids)))
        selected = pred[pred.window_id == window].sort_values("forecast_step")
        long = selected.melt(id_vars=["forecast_step"], value_vars=["actual", "predicted"],
                             var_name="series", value_name="value")
        fig = px.line(long, x="forecast_step", y="value", color="series", markers=True,
                      title=f"{pred_site} · {SENSOR_NAMES.get(pred_sensor, pred_sensor)} · window {window}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(selected, use_container_width=True, hide_index=True)
    else:
        st.warning("Unexpected prediction CSV format.")
else:
    st.warning(f"Prediction file not found: {pred_path}")

st.divider()
st.subheader("FL training log explorer (optional)")
log_upload = st.file_uploader("Upload FL simulation log (.txt/.log)", type=["txt", "log"])
if log_upload:
    log_text = log_upload.getvalue().decode("utf-8", errors="ignore")
    st.text_area("Log preview", log_text[:5000], height=180)
    # Detect common log patterns such as 'Round 3 ... loss: 0.123'
    regex = re.compile(r"(?:round|iteration)\s*[:#]?\s*(\d+).*?(?:loss|train_loss|training_loss)\s*[:=]\s*([0-9.eE+-]+)",
                       re.IGNORECASE | re.DOTALL)
    found = regex.findall(log_text)
    if found:
        log_df = pd.DataFrame(found, columns=["round", "loss"])
        log_df["round"] = pd.to_numeric(log_df["round"])
        log_df["loss"] = pd.to_numeric(log_df["loss"])
        log_df = log_df.drop_duplicates("round").sort_values("round")
        st.line_chart(log_df.set_index("round")["loss"])
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Round/loss pairs weren't detected automatically. Share the log format to adapt the parser.")

st.info("For FL actual-vs-predicted plots, we need to update the evaluation script to export per-point federated predictions.")
