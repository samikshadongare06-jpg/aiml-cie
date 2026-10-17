# Project Context: Soybean Sowing Window Predictor

## Goal
Predict whether it is safe to sow soybean on a given date (and so the best sowing week) for Ahilyanagar district, Maharashtra, India (lat 19.09, lon 74.74), using local weather and climate indices (ONI, DMI/IOD, SOI). College AIML course project, kept simple on purpose.

## Scope
- Crop: soybean (kharif season only)
- Location: Ahilyanagar only (one point). Pooling nearby districts is an optional upgrade if results are weak
- Years: 1995 to 2025
- Model: Random Forest classifier (scikit-learn), compared with Decision Tree and Logistic Regression
- Deliverables: a Flask website with two tabs (Predict, Our Process) and a notebook with the experiments

## Tech stack and environment
- Python 3.10+, local virtual environment (venv), VS Code, local Jupyter (no Colab)
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, flask, joblib, requests, jupyter
- Frontend: plain HTML, CSS and JavaScript with Chart.js (no React)
- Version control: GitHub (GitHub Desktop). venv/ is in .gitignore and is never shared

## Folder structure
```
sowing-predictor/
  CONTEXT.md
  requirements.txt
  app.py
  venv/             local only, never committed
  data/raw/         weather_daily.csv, oni.csv, dmi.csv, soi.csv, kerala_onset.csv (optional)
  data/processed/   weather_features.csv, final_dataset.csv, annual_dataset.csv
  src/              01_download_weather.py, 02_prepare_indices.py, 03_build_features.py,
                    04_make_labels.py, 05_merge_final.py
  notebooks/        experiments.ipynb
  models/           best_model.joblib, features.json
  static/           css/, js/, charts/, results.json
  templates/        index.html
```
Rules:
- Raw data is never edited by hand. Scripts read data/raw and write data/processed.
- Scripts are run from the project root, e.g. `python src/01_download_weather.py`.
- The first cell of experiments.ipynb runs `os.chdir('..')` once per session so relative paths work.
- Each person edits different files to avoid Git conflicts. Only the modeling person edits experiments.ipynb.

## Data sources
- Daily weather: Open-Meteo Historical Weather API (ERA5 / ERA5-Land): rain, Tmax, Tmin, ET0, soil moisture (0-7 cm), relative humidity. April to August of each year.
- ONI: NOAA CPC (3-month Pacific temperature anomaly)
- DMI (Indian Ocean Dipole): NOAA PSL
- SOI: NOAA PSL or CPC
- Optional: Kerala monsoon onset dates (IMD); district yield (ICRISAT or data.gov.in) for validation only, never as a model input

## Decision dates
Fixed calendar dates every year: 18 May, 25 May, 1 Jun, 8 Jun, 15 Jun, 22 Jun, 29 Jun, 6 Jul, 13 Jul (9 rows per year, about 279 rows in total).

## Features (all computed using only data BEFORE the decision date)
rain_7d, rain_14d, rain_cum_may1, rain_days_14d, days_since_rain, tmax_7d, tmin_7d, soil_7d, rh_7d, et0_7d, doy, oni, oni_change, dmi, soi

Climate index timing rule (use only what was already published):
- DMI and SOI: value of the month before the decision month
- ONI: 3-month season centred two months before the decision month

## Label rule (safe_label = 1 if all three are true)
1. Cumulative rain since 1 May is at least CUM_MM (75 mm)
2. Rain in the 21 days starting on the decision date is at least FUT_MM (40 mm)
3. No dry spell of DRY_RUN (10) or more days in those 21 days (a dry day has rain under 2.5 mm)

The constants are tuned so years spread across classes, and justified from MPKV Rahuri or KVK advice.

Year class = first safe decision date:
- 1 Jun or earlier: June_W1
- 8 Jun: June_W2
- 15 Jun: June_W3
- 22 or 29 Jun: June_W4
- 6 Jul or later, or never: July_or_later

year_class_3 merges June_W1, June_W2 and June_W3 into By_mid_June, and June_W4 into Late_June, because the early classes have only 1 to 6 years.

## Final CSV: data/processed/final_dataset.csv
year, decision_date, doy, rain_7d, rain_14d, rain_cum_may1, rain_days_14d, days_since_rain, tmax_7d, tmin_7d, soil_7d, rh_7d, et0_7d, oni, oni_change, dmi, soi, safe_label, year_class

## Modeling rules
- random_state=42 everywhere
- Always compare against a majority-class baseline (DummyClassifier)
- Never split weekly rows randomly: use GroupKFold with groups = year
- Metrics: accuracy, F1, confusion matrix, permutation importance
- Experiment ladder: Exp 0 annual 80/20 split, Exp 1 annual K-fold, Exp 2 weekly with grouped K-fold, Exp 3 feature selection, Exp 4 tuning and model comparison
- Optional: lead-time comparison (feature sets A, B, C) and a global-warming trend chart (analysis only, not a model feature)
- Every result is logged to static/results.json and every chart saved to static/charts/

## Website (demo mode)
- Tab 1 Predict: dropdown "load a past example" (year + decision date) fills the form from final_dataset.csv; the user can also edit values by hand. Result card shows probability, verdict (Safe to sow / Wait one more week / Not safe: dry break likely), the top 3 features with the entered value next to the historical median, and in demo mode what actually happened that year.
- Tab 2 Our Process: scrolling story of sources, integration, features, label rule, experiment ladder, importance, lead-time chart, climate trend, limitations and future work.
- Flask routes: `/`, `/predict` (POST JSON), `/example`
- Palette: #1B4332, #2D6A4F, #95D5B2, #F6FBF7, #E9A820, #C8553D. Font: Poppins or Inter. Chart.js file stored locally in static/js so the demo works offline.

## How we describe deployment (be honest)
This course demo replays historical cases to compare predictions with what actually happened. In a deployed version, the farmer would pick a date and enter only case-specific details; weather values would be auto-filled from a trusted source (Open-Meteo recent weather) and climate indices from NOAA monthly values. This is future work and is NOT built. Never claim it works live.

## Instructions for the AI assistant
- Write small functions and show a quick check using df.head()
- Explain each block in plain English, as if to a beginner
- Use pandas and scikit-learn only; keep code simple and commented
- Use paths relative to the project root
- Never use data from after the decision date as a feature
- If something in this file seems wrong or unclear, say so instead of guessing

## Known limitations (say these in the viva before being asked)
- Rainfall is ERA5 reanalysis at one point (19.09 N, 74.74 E), not gauge data, and can differ from local reports.
- Labels come from our own rule, not official year-by-year sowing dates. The rule rewards rain and has no waterlogging ceiling.
- 2025: the rule marks 25 May as safe after unusually heavy early rain, while the Maharashtra Agriculture Department advised farmers to wait.
- Validation reference: MPKV Rahuri (Gramin Krishi Mausam Sewa) agromet bulletins for Ahilyanagar.