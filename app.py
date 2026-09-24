"""House Price Predictor – Streamlit app for houseprice_pipeline.pkl

Run:  streamlit run app.py
Needs: streamlit, scikit-learn, pandas, numpy, joblib
(use the same scikit-learn version the model was trained with, see requirements.txt note below)

The pickled pipeline contains ONLY a RandomForestRegressor (no encoder), so this app
builds the exact 87 one-hot / numeric columns the model was trained on.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "houseprice_pipeline.pkl"

# Category that was dropped by get_dummies(drop_first=True) -> all its dummy columns = 0.
# (Inferred from the column list; change the labels if your raw data differs.)
BASELINES = {
    "newlocation_": "agra",
    "Furnishing_": "Furnished",
    "Transaction_": "New Property",
    "Ownership_": "Co-operative Society",
    "facing_": "East",
}
OVERLOOKING = {
    "Pool": "overlooking_Pool",
    "Garden / Park": "overlooking_Garden_Park",
    "Main Road": "overlooking_Main_Road",
}

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    return joblib.load(MODEL_PATH)


def options_for(features, prefix):
    """Category labels for a one-hot group, with the dropped baseline first."""
    labels = [f[len(prefix):] for f in features if f.startswith(prefix)]
    return [BASELINES[prefix]] + labels


def build_row(features, values):
    """Return a 1-row DataFrame in the model's exact column order."""
    row = dict.fromkeys(features, 0)
    row.update({k: v for k, v in values.items() if k in row})
    return pd.DataFrame([row], columns=list(features))


def inr(x):
    if x >= 1e7:
        return f"₹ {x / 1e7:,.2f} Crore"
    if x >= 1e5:
        return f"₹ {x / 1e5:,.2f} Lakh"
    return f"₹ {x:,.0f}"


# ------------------------------------------------------------------ UI
st.title("🏠 House Price Predictor")
st.caption("Random Forest model · enter the property details and get an estimated price.")

try:
    pipe = load_model()
except Exception as e:  # missing file, sklearn version mismatch, etc.
    st.error(f"Could not load `{MODEL_PATH.name}`: {e}")
    st.stop()

FEATURES = list(pipe.feature_names_in_) if hasattr(pipe, "feature_names_in_") else \
    list(pipe.named_steps["model"].feature_names_in_)

with st.form("inputs"):
    c1, c2 = st.columns(2)
    with c1:
        location = st.selectbox("City / Location", options_for(FEATURES, "newlocation_"))
        carpet_area = st.number_input("Carpet area (sq ft)", 100, 20000, 1000, step=50)
        bathrooms = st.number_input("Bathrooms", 1, 10, 2)
        balcony = st.number_input("Balconies", 0, 10, 1)
        furnishing = st.selectbox("Furnishing", options_for(FEATURES, "Furnishing_"))
    with c2:
        total_floors = st.number_input("Total floors in building", 1, 100, 10)
        floor_number = st.number_input("Floor number", 0, 100, 3)
        transaction = st.selectbox("Transaction type", options_for(FEATURES, "Transaction_"))
        ownership = st.selectbox("Ownership", options_for(FEATURES, "Ownership_"))
        facing = st.selectbox("Facing", options_for(FEATURES, "facing_"))
    overlooking = st.multiselect("Overlooking", list(OVERLOOKING))
    submitted = st.form_submit_button("Predict price", type="primary", use_container_width=True)

if submitted:
    if floor_number > total_floors:
        st.error("Floor number can't be higher than total floors.")
        st.stop()

    values = {
        "Carpet Area": carpet_area,
        "Bathroom": bathrooms,
        "Balcony": balcony,
        "Floor_Number": floor_number,
        "Total_Floors": total_floors,
    }
    # One-hot groups: a baseline choice simply leaves every dummy at 0.
    for prefix, choice in [
        ("newlocation_", location),
        ("Furnishing_", furnishing),
        ("Transaction_", transaction),
        ("Ownership_", ownership),
        ("facing_", facing),
    ]:
        values[prefix + choice] = 1
    for label in overlooking:
        values[OVERLOOKING[label]] = 1

    X = build_row(FEATURES, values)
    try:
        price = float(pipe.predict(X)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.success("Estimated price")
    st.metric(label="Predicted price", value=inr(price))
    st.caption(f"Raw model output: {price:,.0f}  (assumed INR – adjust `inr()` if your target uses other units)")

    with st.expander("Model input (debug)"):
        st.dataframe(X.loc[:, (X != 0).any()].T.rename(columns={0: "value"}))