#  Heat-Safe Bangladesh
**An Air Quality and Weather-Based Outdoor Activity Planner**

An end-to-end data science project that analyzes 8 years of hourly air-quality and weather data from Dhaka,
Bangladesh to identify the safest times for outdoor activity, and builds machine learning models that give
advance warning of worsening air quality.

>  General outdoor-planning guidance only — not medical advice.

## What it does
- Cleans and explores ~8 years of hourly pollutant (SO2, NO, NO2, NOX, CO, O3, PM2.5, PM10) and weather
  (temperature, humidity, solar radiation) data
- Answers key questions: which hours/months have the worst air quality, which pollutants drive the AQI, and how
  weather relates to pollution levels
- Engineers a transparent, rule-based **Outdoor Risk** score (Low / Moderate / High / Extreme) combining AQI
  severity with heat and humidity
- Trains and compares three classifiers (Logistic Regression, Decision Tree, Random Forest) to **forecast the
  next recorded period's AQI severity category**, with explicit safeguards against data leakage
  (chronological train/test split, no use of the current AQI value as a feature)
- Ships an optional **Streamlit dashboard** for interactively exploring trends and predictions

## Key findings
- Late afternoon/early evening hours (~3–8 PM) tend to be the safest window for outdoor activity in Dhaka;
  mid-morning (~9 AM–12 PM) tends to be riskiest
- Winter months (Nov–Feb) are far more polluted than the monsoon season (Jun–Sep)
- PM2.5 and PM10 are the dominant drivers of AQI, followed by traffic-related gases (NO, NOX)
- Random Forest gave the best balance of precision/recall across all severity classes (evaluated with macro
  F1-score due to class imbalance)

## Tech stack
Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Streamlit

## Repo contents
| File | Description |
|---|---|
| `heat_safe_bangladesh.ipynb` | Full analysis notebook: cleaning, EDA, feature engineering, modeling, evaluation |
| `app.py` | Optional Streamlit dashboard |
| `cleaned_AQI.csv` | Source dataset (hourly AQI/weather readings, multiple Bangladeshi cities) |

## Getting started
```bash
git clone <your-repo-url>
cd heat-safe-bangladesh
pip install pandas numpy matplotlib seaborn scikit-learn jupyter streamlit

# run the notebook
jupyter notebook heat_safe_bangladesh.ipynb

# or launch the dashboard
streamlit run app.py
```

## Notes on the data
The dataset includes 12 Bangladeshi cities; this project focuses on Dhaka (the capital, with the longest and most
complete hourly record) as a representative, beginner-friendly case study. The same pipeline can be adapted to
other cities.

## License
MIT (or update as needed).
