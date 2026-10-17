import pandas as pd
import numpy as np

# ---------- load the pieces ----------
feats = pd.read_csv('data/processed/weather_features.csv', parse_dates=['decision_date'])
labels = pd.read_csv('data/processed/labels.csv', parse_dates=['decision_date'])
yc = pd.read_csv('data/processed/year_classes.csv')

def load_index(name):
    df = pd.read_csv(f'data/raw/{name}.csv')
    df['date'] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    return df.set_index('date')['value']

oni, dmi, soi = load_index('oni'), load_index('dmi'), load_index('soi')

def lagged(series, d, months_back):
    """Value of an index 'months_back' months before the month of date d."""
    key = d.replace(day=1) - pd.DateOffset(months=months_back)
    return series.get(key, np.nan)

# ---------- attach climate indices using only what was already published ----------
feats['oni'] = [lagged(oni, d, 2) for d in feats.decision_date]                             # season centred 2 months earlier
feats['oni_change'] = [lagged(oni, d, 2) - lagged(oni, d, 5) for d in feats.decision_date]  # change over 3 months
feats['dmi'] = [lagged(dmi, d, 1) for d in feats.decision_date]                             # previous month
feats['soi'] = [lagged(soi, d, 1) for d in feats.decision_date]                             # previous month

# ---------- attach labels ----------
final = feats.merge(labels[['year', 'decision_date', 'safe_label']], on=['year', 'decision_date'])
final = final.merge(yc[['year', 'year_class', 'year_class_3']], on='year')

COLS = ['year', 'decision_date', 'doy', 'rain_7d', 'rain_14d', 'rain_cum_may1', 'rain_days_14d',
        'days_since_rain', 'tmax_7d', 'tmin_7d', 'soil_7d', 'rh_7d', 'et0_7d',
        'oni', 'oni_change', 'dmi', 'soi', 'safe_label', 'year_class','year_class_3']
final = final[COLS]
final.to_csv('data/processed/final_dataset.csv', index=False)

print('final_dataset:', final.shape, '(expect 279 rows x 20 columns)')
print('columns with blank cells (should print nothing below):')
print(final.isna().sum()[final.isna().sum() > 0])
print(final.head())

# ---------- annual table for Experiments 0 and 1 (one row per year, issued on 1 June) ----------
w = pd.read_csv('data/raw/weather_daily.csv', parse_dates=['date']).sort_values('date').set_index('date')
annual = []
for year in range(1995, 2026):
    am = w.loc[f'{year}-04-01': f'{year}-05-31']
    may = w.loc[f'{year}-05-01': f'{year}-05-31']
    d = pd.Timestamp(f'{year}-06-01')
    annual.append({
        'year': year,
        'rain_apr_may': am['precipitation_sum'].sum(),
        'rain_days_apr_may': int((am['precipitation_sum'] >= 2.5).sum()),
        'tmax_apr_may': am['temperature_2m_max'].mean(),
        'soil_may': may['soil_moisture_0_to_7cm'].mean(),
        'rh_may': may['relative_humidity_2m'].mean(),
        'oni_mam': lagged(oni, d, 2),
        'dmi_may': lagged(dmi, d, 1),
        'soi_may': lagged(soi, d, 1),
    })
annual = pd.DataFrame(annual).merge(yc[['year', 'year_class', 'year_class_3']], on='year')
annual.to_csv('data/processed/annual_dataset.csv', index=False)
print('\nannual_dataset:', annual.shape, '(expect 31 rows x 11 columns)')
print('blank cells:', int(annual.isna().sum().sum()))