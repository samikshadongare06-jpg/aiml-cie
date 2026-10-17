import requests, pandas as pd

URLS = {
    'oni': 'https://psl.noaa.gov/data/correlation/oni.data',
    'soi': 'https://psl.noaa.gov/data/correlation/soi.data',
    'dmi': 'https://psl.noaa.gov/data/timeseries/month/data/dmi.had.long.data',
}

def parse(text):
    rows = []
    for line in text.splitlines():
        p = line.split()
        # a data line = a year followed by 12 monthly values
        if len(p) == 13 and p[0].isdigit() and 1950 <= int(p[0]) <= 2030:
            for month, v in enumerate(p[1:], start=1):
                if float(v) > -90:              # -99.99 or -9999 means missing
                    rows.append((int(p[0]), month, float(v)))
    df = pd.DataFrame(rows, columns=['year', 'month', 'value'])
    return df[(df.year >= 1994) & (df.year <= 2025)]

for name, url in URLS.items():
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        open(f'data/raw/{name}_raw.txt', 'w').write(r.text)   # keep the original file
        df = parse(r.text)
        df.to_csv(f'data/raw/{name}.csv', index=False)
        print(name, df.shape)
    except Exception as e:
        print(name, 'FAILED:', e)