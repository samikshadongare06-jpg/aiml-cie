import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
model = joblib.load('models/best_model.joblib')
FEATURES = json.load(open('models/features.json'))
MEDIANS = json.load(open('models/medians.json'))
IMPORTANCES = json.load(open('models/importances.json'))
DATA = pd.read_csv('data/processed/final_dataset.csv')
CUTOFF, RULE_MM = 0.5, 75

# label, unit, hint for each form box
INFO = {
    'rain_cum_may1': ('Rain since 1 May', 'mm', 'Total rain from 1 May to yesterday'),
    'rain_7d': ('Rain, last 7 days', 'mm', 'Total rain in the past week'),
    'rain_days_14d': ('Rainy days, last 14', 'days', 'Days with 2.5 mm or more'),
    'days_since_rain': ('Days since rain', 'days', 'Days since the last 2.5 mm day'),
    'soil_7d': ('Topsoil moisture', 'm3/m3', 'Average of the past week (0.05 to 0.45)'),
    'et0_7d': ('Evaporation demand', 'mm/day', 'Average of the past week'),
    'tmin_7d': ('Night temperature', 'deg C', 'Average low of the past week'),
    'oni': ('ONI', 'index', 'Pacific warmth, two months earlier'),
    'oni_change': ('ONI change', 'index', 'Change over three months'),
    'soi': ('SOI', 'index', 'Pacific pressure index, last month'),
}


@app.route('/')
def home():
    feats = [{'key': f, 'label': INFO[f][0], 'unit': INFO[f][1], 'hint': INFO[f][2], 'median': MEDIANS[f]}
             for f in FEATURES]
    mds = sorted(DATA['decision_date'].str[5:10].unique())
    dates = [{'v': m, 'label': pd.Timestamp('2000-' + m).strftime('%d %b')} for m in mds]
    return render_template('index.html', features=feats, years=sorted(DATA['year'].unique()), dates=dates)


@app.route('/example')
def example():
    row = DATA[DATA['decision_date'].str.startswith(f"{request.args.get('year')}-{request.args.get('date')}")]
    if row.empty:
        return jsonify({'error': 'No such case'})
    row = row.iloc[0]
    return jsonify({'values': {f: float(row[f]) for f in FEATURES}, 'actual': int(row['safe_label'])})


@app.route('/predict', methods=['POST'])
def predict():
    d = request.get_json(silent=True) or {}
    try:
        vals = {f: float(d[f]) for f in FEATURES}
    except (KeyError, ValueError, TypeError):
        return jsonify({'error': 'Fill every box with a number.'})
    p = float(model.predict_proba(pd.DataFrame([vals])[FEATURES])[0][1])
    top = sorted(IMPORTANCES, key=IMPORTANCES.get, reverse=True)[:3]
    return jsonify({
        'safe': p >= CUTOFF, 'score': round(p * 100), 'rain': vals['rain_cum_may1'], 'rule_mm': RULE_MM,
        'top': [{'label': INFO[f][0], 'unit': INFO[f][1], 'value': vals[f], 'median': MEDIANS[f]} for f in top],
    })


if __name__ == '__main__':
    app.run(debug=True)
