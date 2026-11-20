import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import FEATURES, N_CYCLES, features_from_cycle_file

st.set_page_config(page_title="Battery Cycle Life Predictor", page_icon="🔋")


@st.cache_resource
def load():
    model = joblib.load("models/tuned_rf.joblib")
    meta = json.load(open("models/model_meta.json"))
    return model, meta


model, meta = load()
ranges = meta["feature_ranges"]  # feature -> [min, max, median]
hm = meta["heldout_metrics"]


def predict(row: pd.DataFrame):
    """Point prediction plus the spread across the forest's trees."""
    imputed = model.named_steps["imputer"].transform(row[FEATURES])
    forest = model.named_steps["regressor"]
    per_tree = np.array([t.predict(imputed)[0] for t in forest.estimators_])
    return float(per_tree.mean()), float(np.percentile(per_tree, 10)), float(np.percentile(per_tree, 90))


def show_result(row: pd.DataFrame):
    mean, lo, hi = predict(row)
    st.metric("Predicted total cycle life", f"{mean:,.0f} cycles")
    st.caption(
        f"Tree-to-tree spread (10th to 90th percentile): {lo:,.0f} to {hi:,.0f} cycles. "
        f"Typical test error of this model is about ±{hm['MAE']:.0f} cycles (MAE)."
    )
    out_of_range = [
        f for f in FEATURES
        if pd.notna(row[f].iloc[0]) and not (ranges[f][0] <= row[f].iloc[0] <= ranges[f][1])
    ]
    if out_of_range:
        st.warning(
            "These values are outside the range seen in training, so the prediction "
            "may be unreliable: " + ", ".join(out_of_range)
        )


st.title("🔋 Battery Cycle Life Predictor")
st.write(
    f"Predicts a battery's **total cycle life** from its **first {N_CYCLES} cycles**, "
    "using a tuned Random Forest."
)

tab_upload, tab_manual, tab_about = st.tabs(["Upload cycle data", "Enter values", "About model"])

with tab_upload:
    st.write(
        f"Upload a CSV for **one battery** with at least {N_CYCLES} cycles (one row per cycle). "
        f"Required columns: `{'`, `'.join(FEATURES)}`. Optional: `cycle_index`."
    )
    up = st.file_uploader("Cycle CSV", type="csv")
    if up is not None:
        try:
            cycles = pd.read_csv(up)
            row = features_from_cycle_file(cycles)
            st.subheader("Features (mean of first 50 cycles)")
            st.dataframe(row.T.rename(columns={0: "value"}))
            show_result(row)
        except Exception as e:
            st.error(str(e))

with tab_manual:
    st.write(f"Enter each feature as its **mean over the first {N_CYCLES} cycles**.")
    vals = {}
    for f in FEATURES:
        lo, hi, med = ranges[f]
        vals[f] = st.number_input(f, value=float(med), format="%.6f",
                                  help=f"Training range: {lo:.4g} to {hi:.4g}")
    if st.button("Predict", type="primary"):
        show_result(pd.DataFrame([vals]))

with tab_about:
    st.write("**Model:** Random Forest (400 trees, min_samples_leaf=4, min_samples_split=5, max_features=0.8) with median imputation.")
    st.write(f"**Held-out test results ({hm['n_test']} batteries):** MAE {hm['MAE']:.2f}, RMSE {hm['RMSE']:.2f}, R² {hm['R2']:.4f}")
    st.write(f"**Final model** is refit on all {meta['n_batteries_final_fit']} batteries with the same hyperparameters.")
    st.write(f"**Features:** {', '.join(FEATURES)}")
    st.write(f"**Training cycle life range:** {meta['target_range'][0]:.0f} to {meta['target_range'][1]:.0f} cycles")
    st.info("R² of about 0.61 means the model explains roughly 60% of the variance. Use it for screening, not as a precise guarantee.")
