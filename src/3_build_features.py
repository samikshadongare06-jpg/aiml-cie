import pandas as pd
import numpy as np

DATES = ['05-18', '05-25', '06-01', '06-08', '06-15', '06-22', '06-29', '07-06', '07-13']

# ---------- load and clean ----------
w = pd.read_csv('data/raw/weather_daily.csv', parse_dates=['date'])
w = w.sort_values('date').drop_duplicates('date').set_index('date')

print('rows:', len(w), '(expect about 4743 = 31 years x 153 days)')
print('missing values per column:')
print(w.isna().sum())

w = w.interpolate(limit=3)                                     # fill tiny gaps only
w['precipitation_sum'] = w['precipitation_sum'].clip(lower=0)  # rain cannot be negative

# ---------- build features, one row per year and decision date ----------
rows = []
for year in range(1995, 2026):
    for md in DATES:
        d = pd.Timestamp(f'{year}-{md}')
        # ONLY this year's data from 1 April up to the day BEFORE the decision date
        past = w.loc[f'{year}-04-01': d - pd.Timedelta(days=1)]
        rain = past['precipitation_sum']
        wet = (rain >= 2.5).values                       # a "rainy day" = 2.5 mm or more
        last_wet = np.nonzero(wet)[0]
        rows.append({
            'year': year,
            'decision_date': d,
            'doy': d.dayofyear,
            'rain_7d': rain.tail(7).sum(),
            'rain_14d': rain.tail(14).sum(),
            'rain_cum_may1': rain.loc[f'{year}-05-01':].sum(),
            'rain_days_14d': int(wet[-14:].sum()),
            'days_since_rain': (len(wet) - 1 - last_wet[-1]) if len(last_wet) else 60,
            'tmax_7d': past['temperature_2m_max'].tail(7).mean(),
            'tmin_7d': past['temperature_2m_min'].tail(7).mean(),
            'soil_7d': past['soil_moisture_0_to_7cm'].tail(7).mean(),
            'rh_7d': past['relative_humidity_2m'].tail(7).mean(),
            'et0_7d': past['et0_fao_evapotranspiration'].tail(7).mean(),
        })

feats = pd.DataFrame(rows)
feats.to_csv('data/processed/weather_features.csv', index=False)

print('\nshape:', feats.shape, '(expect 279 rows x 13 columns)')
print('blank cells:', int(feats.isna().sum().sum()), '(expect 0)')
print(feats.head(9))