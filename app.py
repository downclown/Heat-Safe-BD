"""
Heat-Safe Bangladesh — Streamlit Dashboard
============================================
Companion app for the "Heat-Safe Bangladesh" notebook.

Run locally with:
    pip install streamlit scikit-learn pandas numpy matplotlib seaborn
    streamlit run app.py

Expects `cleaned_AQI.csv` (the original dataset) to be in the same folder,
or update CSV_PATH below.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Heat-Safe Bangladesh", page_icon="🌤️", layout="wide")

CSV_PATH = "cleaned_AQI.csv"
RANDOM_STATE = 42

POLLUTANT_COLS = ["SO2", "NO", "NO2", "NOX", "CO", "O3", "PM2.5", "PM10"]
WEATHER_COLS = ["Temperature", "RH", "Solar Rad"]
NUMERIC_COLS = POLLUTANT_COLS + WEATHER_COLS + ["Month", "Day", "AQI"]
SEVERITY_ORDER = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
RISK_ORDER = ["Low", "Moderate", "High", "Extreme"]
RISK_COLORS = {"Low": "#2ecc71", "Moderate": "#f1c40f", "High": "#e67e22", "Extreme": "#c0392b"}


def aqi_to_severity(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def temperature_category(t):
    if t < 20:
        return "Cool"
    elif t < 26:
        return "Mild"
    elif t < 32:
        return "Warm"
    return "Hot"


def humidity_category(h):
    if h < 40:
        return "Low"
    elif h < 60:
        return "Moderate"
    elif h < 80:
        return "High"
    return "Very High"


def risk_bucket(score):
    if score <= 1:
        return "Low"
    elif score <= 3:
        return "Moderate"
    elif score <= 5:
        return "High"
    return "Extreme"


@st.cache_data(show_spinner="Loading and cleaning data (first run only, then cached)...")
def load_and_clean(csv_path):
    df_raw = pd.read_csv(csv_path)
    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["Location"] == "Dhaka"].copy()

    is_midnight_24 = df["Time"].astype(str).str.strip() == "24:00"
    time_clean = df["Time"].astype(str).str.strip().replace({"24:00": "00:00"})
    date_part = pd.to_datetime(df["Date"], format="mixed").dt.normalize()
    date_part.loc[is_midnight_24] = date_part.loc[is_midnight_24] + pd.Timedelta(days=1)
    df["DateTime"] = pd.to_datetime(date_part.dt.strftime("%Y-%m-%d") + " " + time_clean, errors="coerce")

    df = df.groupby("DateTime", as_index=False)[NUMERIC_COLS].mean()

    df.loc[df["SO2"] < 0, "SO2"] = np.nan
    df.loc[(df["Temperature"] < 5) | (df["Temperature"] > 45), "Temperature"] = np.nan
    df.loc[df["RH"] > 100, "RH"] = 100.0

    df = df.sort_values("DateTime").reset_index(drop=True)
    df[NUMERIC_COLS] = df[NUMERIC_COLS].interpolate(method="linear", limit_direction="both")

    df["AQI_severity"] = df["AQI"].apply(aqi_to_severity)
    df["Year"] = df["DateTime"].dt.year
    df["Month"] = df["DateTime"].dt.month
    df["Day"] = df["DateTime"].dt.day
    df["Hour"] = df["DateTime"].dt.hour
    df["Day_of_week"] = df["DateTime"].dt.day_name()
    df["Is_weekend"] = df["Day_of_week"].isin(["Friday", "Saturday"]).astype(int)
    df["Temp_category"] = df["Temperature"].apply(temperature_category)
    df["Humidity_category"] = df["RH"].apply(humidity_category)

    pm25_spike_threshold = df["PM2.5"].quantile(0.90)
    df["Pollution_spike"] = (df["PM2.5"] > pm25_spike_threshold).astype(int)
    df["Prev_AQI"] = df["AQI"].shift(1)
    df["Rolling_AQI_24h"] = df["AQI"].shift(1).rolling(window=24, min_periods=6).mean()

    severity_score_map = {"Good": 0, "Moderate": 1, "Unhealthy for Sensitive Groups": 2,
                           "Unhealthy": 3, "Very Unhealthy": 4, "Hazardous": 5}
    df["severity_score"] = df["AQI_severity"].map(severity_score_map)
    solar_p75 = df["Solar Rad"].quantile(0.75)

    def weather_penalty(row):
        p = 0
        if row["Temp_category"] == "Hot":
            p += 1
        if row["Humidity_category"] in ("High", "Very High"):
            p += 1
        if row["Solar Rad"] > solar_p75:
            p += 1
        return p

    df["weather_penalty"] = df.apply(weather_penalty, axis=1)
    df["risk_score"] = df["severity_score"] + df["weather_penalty"]
    df["Outdoor_Risk"] = df["risk_score"].apply(risk_bucket)

    return df


@st.cache_resource(show_spinner="Training the next-period AQI severity model (first run only, then cached)...")
def train_model(df):
    df = df.sort_values("DateTime").reset_index(drop=True)
    df["AQI_severity_next"] = df["AQI_severity"].shift(-1)
    model_df = df.dropna(subset=["AQI_severity_next", "Prev_AQI", "Rolling_AQI_24h"]).copy()

    numeric_features = POLLUTANT_COLS + WEATHER_COLS + ["Prev_AQI", "Rolling_AQI_24h", "Hour", "Month"]
    categorical_features = ["Temp_category", "Humidity_category", "Day_of_week"]
    binary_features = ["Is_weekend", "Pollution_spike"]
    feature_cols = numeric_features + categorical_features + binary_features

    X = model_df[feature_cols]
    y = model_df["AQI_severity_next"]
    split_idx = int(len(model_df) * 0.8)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ], remainder="passthrough")

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                          random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)
    return pipe, feature_cols, model_df


# ------------------------------------------------------------------
# App layout
# ------------------------------------------------------------------
st.title("🌤️ Heat-Safe Bangladesh")
st.caption("Air Quality and Weather-Based Outdoor Activity Planner — Dhaka, Bangladesh")
st.warning("General outdoor-planning guidance only — this is **not medical advice**.", icon="⚠️")

try:
    df = load_and_clean(CSV_PATH)
except FileNotFoundError:
    st.error(f"Couldn't find `{CSV_PATH}`. Place `cleaned_AQI.csv` next to `app.py` and rerun.")
    st.stop()

pipe, feature_cols, model_df = train_model(df)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Trends", "🔮 Predict next period", "🧭 Outdoor Risk lookup", "✅ Safest hours"]
)

# --- Tab 1: Trends ---
with tab1:
    st.subheader("Pick a date to explore")
    min_date, max_date = df["DateTime"].dt.date.min(), df["DateTime"].dt.date.max()
    picked_date = st.date_input("Date", value=max_date, min_value=min_date, max_value=max_date)

    day_data = df[df["DateTime"].dt.date == picked_date]
    if day_data.empty:
        st.info("No recorded data for that date — try another nearby date.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(day_data["Hour"], day_data["AQI"], marker="o", color="firebrick")
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("AQI")
            ax.set_title(f"Hourly AQI on {picked_date}")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(day_data["Hour"], day_data["Temperature"], marker="o", label="Temperature (°C)", color="orange")
            ax.plot(day_data["Hour"], day_data["RH"], marker="o", label="Humidity (%)", color="teal")
            ax.set_xlabel("Hour of day")
            ax.legend()
            ax.set_title(f"Temperature & Humidity on {picked_date}")
            st.pyplot(fig)

        st.dataframe(
            day_data[["Hour", "AQI", "AQI_severity", "Temperature", "RH", "Solar Rad", "Outdoor_Risk"]]
            .reset_index(drop=True)
        )

# --- Tab 2: Predict next period ---
with tab2:
    st.subheader("Predict the next recorded period's AQI severity")
    st.write(
        "Pick a recorded hour below; the model uses that hour's pollutant/weather readings plus recent AQI "
        "history to forecast the **next recorded period's** AQI severity category."
    )
    options = model_df["DateTime"].dt.strftime("%Y-%m-%d %H:%M")
    picked = st.selectbox("Choose a recorded hour", options=options.iloc[::-1])
    row = model_df[model_df["DateTime"].dt.strftime("%Y-%m-%d %H:%M") == picked].iloc[[0]]

    pred = pipe.predict(row[feature_cols])[0]
    proba = pipe.predict_proba(row[feature_cols])[0]
    classes = pipe.named_steps["model"].classes_

    st.metric("Predicted next-period AQI severity", pred)
    proba_df = pd.DataFrame({"Severity": classes, "Probability": proba}).sort_values("Probability", ascending=False)
    st.bar_chart(proba_df.set_index("Severity"))

    st.caption(
        "This model was trained only on Dhaka data (2012–2020) and does not use the current hour's own AQI "
        "value, to avoid the model trivially copying the current reading forward."
    )

# --- Tab 3: Outdoor Risk lookup ---
with tab3:
    st.subheader("Outdoor Risk for a chosen hour")
    row2 = row if 'row' in dir() else model_df.iloc[[-1]]
    st.metric("Outdoor Risk category", row2["Outdoor_Risk"].values[0])
    st.write(
        f"- AQI severity: **{row2['AQI_severity'].values[0]}**\n"
        f"- Temperature: **{row2['Temperature'].values[0]:.1f} °C** ({row2['Temp_category'].values[0]})\n"
        f"- Humidity: **{row2['RH'].values[0]:.1f}%** ({row2['Humidity_category'].values[0]})\n"
        f"- Solar radiation: **{row2['Solar Rad'].values[0]:.1f} W/m²**"
    )
    st.caption("Outdoor Risk = AQI-severity score + weather penalty. See the notebook, section 6, for the full rule.")

# --- Tab 4: Safest hours ---
with tab4:
    st.subheader("Historically safest hours for outdoor activity (Dhaka)")
    hourly_profile = df.groupby("Hour").agg(
        avg_risk_score=("risk_score", "mean"),
        avg_AQI=("AQI", "mean"),
        avg_Temperature=("Temperature", "mean"),
        avg_RH=("RH", "mean"),
    ).round(2).sort_values("avg_risk_score")

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=hourly_profile.index, y=hourly_profile["avg_risk_score"], palette="RdYlGn_r", ax=ax)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average risk score (lower = safer)")
    st.pyplot(fig)

    st.dataframe(hourly_profile)
    st.success(
        f"✅ Safest hour on average: **{hourly_profile.index[0]}:00** "
        f"| ⚠️ Riskiest hour on average: **{hourly_profile.index[-1]}:00**"
    )
