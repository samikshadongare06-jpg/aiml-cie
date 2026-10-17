import requests, pandas as pd, time

LAT, LON = 19.09, 74.74
rows = []
for year in range(1995, 2026):
    r = requests.get('https://archive-api.open-meteo.com/v1/archive', params={
        'latitude': LAT, 'longitude': LON,
        'start_date': f'{year}-04-01', 'end_date': f'{year}-08-31',
        'daily': 'precipitation_sum,temperature_2m_max,temperature_2m_min,et0_fao_evapotranspiration',
        'hourly': 'soil_moisture_0_to_7cm,relative_humidity_2m',
        'timezone': 'Asia/Kolkata'}, timeout=60)
    js = r.json()
    d = pd.DataFrame(js['daily']).rename(columns={'time': 'date'})
    h = pd.DataFrame(js['hourly'])
    h['date'] = h['time'].str[:10]
    h = h.groupby('date', as_index=False).mean(numeric_only=True)
    rows.append(d.merge(h, on='date', how='left'))
    time.sleep(1)
pd.concat(rows).to_csv('data/raw/weather_daily.csv', index=False)