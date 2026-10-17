import pandas as pd

# ---- the rule: tune these three numbers, and justify them in the viva ----
CUM_MM = 75     # rain since 1 May must reach this before sowing
FUT_MM = 40     # rain needed in the 21 days after sowing
DRY_RUN = 10    # a dry spell this long (days) after sowing means failure

DATES = ['05-18', '05-25', '06-01', '06-08', '06-15', '06-22', '06-29', '07-06', '07-13']
CLASS_OF = {
    '05-18': 'June_W1', '05-25': 'June_W1', '06-01': 'June_W1',
    '06-08': 'June_W2', '06-15': 'June_W3',
    '06-22': 'June_W4', '06-29': 'June_W4',
    '07-06': 'July_or_later', '07-13': 'July_or_later',
}

MERGE_3 = {'June_W1': 'By_mid_June', 'June_W2': 'By_mid_June', 'June_W3': 'By_mid_June',
           'June_W4': 'Late_June', 'July_or_later': 'July_or_later'}

w = pd.read_csv('data/raw/weather_daily.csv', parse_dates=['date']).sort_values('date').set_index('date')
rain_all = w['precipitation_sum'].clip(lower=0).fillna(0)

rows = []
for year in range(1995, 2026):
    for md in DATES:
        d = pd.Timestamp(f'{year}-{md}')
        past = rain_all.loc[f'{year}-05-01': d - pd.Timedelta(days=1)]
        future = rain_all.loc[d: d + pd.Timedelta(days=20)]       # 21 days from the decision date

        longest = run = 0                                          # longest run of dry days
        for is_wet in (future >= 2.5).values:
            run = 0 if is_wet else run + 1
            longest = max(longest, run)

        safe = int(past.sum() >= CUM_MM and future.sum() >= FUT_MM and longest < DRY_RUN)
        rows.append({'year': year, 'decision_date': d, 'cum_rain': round(past.sum(), 1),
                     'future_rain': round(future.sum(), 1), 'longest_dry_days': longest,
                     'safe_label': safe})

labels = pd.DataFrame(rows)
labels.to_csv('data/processed/labels.csv', index=False)

# ---- year class = first safe decision date ----
yc = []
for year, g in labels.groupby('year'):
    safe = g[g.safe_label == 1]
    if len(safe):
        first = safe.decision_date.min()
        yc.append({'year': year, 'first_safe_date': first.strftime('%m-%d'),
                   'year_class': CLASS_OF[first.strftime('%m-%d')]})
    else:
        yc.append({'year': year, 'first_safe_date': 'none', 'year_class': 'July_or_later'})
yc = pd.DataFrame(yc)
yc['year_class_3'] = yc['year_class'].map(MERGE_3)
yc.to_csv('data/processed/year_classes.csv', index=False)

print('rows with safe_label = 1:', round(100 * labels.safe_label.mean(), 1), '%')
print('\nYEARS PER CLASS (spread them out by tuning CUM_MM, FUT_MM, DRY_RUN):')
print(yc.year_class.value_counts())
print('\nfirst safe date per year:')
print(yc.to_string(index=False))