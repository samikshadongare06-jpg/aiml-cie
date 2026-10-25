"""
AI-Based Soybean Sowing Window Predictor - Flask application (Ahilyanagar, Maharashtra).

    python app.py                      # http://127.0.0.1:5000

Routes
    GET  /                              dashboard
    GET  /demo                          "Sowing Prediction Demo": frontend-only SIMULATION page (no model, no data, no API)
    GET  /health                        200 when model + dataset are loaded, 503 otherwise
    GET  /api/model-info                training period, test metrics, limitations (read from the saved artifacts)
    GET  /api/historical                years / dates available in historical-demo mode
    GET  /api/historical/<year>/<date>  stored feature row for one historical case (+ what the rainfall rule says in hindsight)
    POST /api/predict                   predict safe / not safe (historical demo or manually entered features)

Only two kinds of input exist: rows from the processed historical dataset, and values typed in by the user.
There is NO live weather or forecast integration - nothing is fetched, invented or imputed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
import feature_utils as fu  # noqa: E402
import inference  # noqa: E402

DEFAULT_CONFIG = {
    "MODEL_PATH": BASE_DIR / "models" / "best_model.joblib",
    "FEATURES_PATH": BASE_DIR / "models" / "features.json",
    "METADATA_PATH": BASE_DIR / "models" / "metadata.json",
    "DATASET_PATH": BASE_DIR / "data" / "processed" / "final_dataset.csv",
    "LABELS_PATH": BASE_DIR / "data" / "processed" / "labels.csv",
    "MAX_CONTENT_LENGTH": 64 * 1024,
}
ALLOWED_PREDICT_KEYS = {"mode", "candidate_date", "features"}

LIMITATIONS = [
    "This is a model estimate of a rainfall-based criterion, not an agronomic recommendation or a guarantee of crop success.",
    "'Safe' means the date satisfies the project's rainfall rule (rain since 1 May, rain in the following 21 days, no long dry "
    "spell). It is not a verified sowing outcome and says nothing about yield.",
    "The rule uses rain AFTER the candidate date, which is unknown in a real decision. The model only sees past weather.",
    "The probability is the classifier's raw output. It has not been calibrated and is not a confidence score.",
    "Weather is ERA5 reanalysis at one grid point near Ahilyanagar, not gauge data.",
]


def _err(status: int, code: str, message: str, details=None):
    body = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return jsonify(body), status


def _py(v):
    """numpy scalar -> plain python (for JSON)."""
    return v.item() if hasattr(v, "item") else v


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
    app.json.sort_keys = False      # keep the feature order of the training pipeline in API responses
    app.config.update(DEFAULT_CONFIG)
    if config:
        app.config.update(config)

    state = {"bundle": None, "model_error": None, "data": None, "data_error": None, "labels": None}

    # ------------------------------------------------------------------ loading
    try:
        state["bundle"] = inference.load_bundle(app.config["MODEL_PATH"], app.config["FEATURES_PATH"],
                                                app.config["METADATA_PATH"])
    except inference.ArtifactError as exc:
        state["model_error"] = str(exc)
        app.logger.error("model not available: %s", exc)
    try:
        df = pd.read_csv(app.config["DATASET_PATH"], parse_dates=["decision_date"])
        fu.raise_if_invalid(df)
        df["date_str"] = df["decision_date"].dt.strftime("%Y-%m-%d")
        state["data"] = df
    except (OSError, ValueError, fu.DataValidationError, pd.errors.ParserError) as exc:
        state["data_error"] = f"historical dataset unavailable: {exc}"
        app.logger.error(state["data_error"])
    try:
        lab = pd.read_csv(app.config["LABELS_PATH"], parse_dates=["decision_date"])
        lab["date_str"] = lab["decision_date"].dt.strftime("%Y-%m-%d")
        state["labels"] = lab
    except Exception:  # optional file
        state["labels"] = None
    app.extensions["sowing_state"] = state

    # ------------------------------------------------------------------ helpers
    def split_of(year: int) -> str:
        b = state["bundle"]
        if not b:
            return "unknown"
        tr = (b.metadata.get("training_period") or {}).get("years")
        te = (b.metadata.get("test_period") or {}).get("years")
        if tr and tr[0] <= year <= tr[1]:
            return "train"
        if te and te[0] <= year <= te[1]:
            return "held_out_test"
        return "unknown"

    def indicators(features: dict, medians: dict | None):
        out = []
        for c in fu.FEATURE_COLUMNS:
            if c == "doy":
                continue
            s = fu.FEATURE_SPECS[c]
            out.append({"key": c, "label": s.label, "unit": s.unit, "value": _py(features[c]),
                        "typical_value": None if medians is None else round(medians[c], 3)})
        return out

    def retrospective(date_str: str, year: int):
        lab = state["labels"]
        if lab is None:
            return None
        m = lab[(lab.year == year) & (lab.date_str == date_str)]
        if m.empty or pd.isna(m.iloc[0]["safe_label"]):
            return None
        r = m.iloc[0]
        return {"rule_label": "safe" if int(r["safe_label"]) == 1 else "not_safe",
                "rain_since_may1_mm": _py(r["cum_rain"]), "rain_next_21_days_mm": _py(r["future_rain"]),
                "longest_dry_spell_days": int(r["longest_dry_days"]),
                "note": "Computed in hindsight from rain AFTER the candidate date; not available when a real decision is made."}

    def explanation(pred: dict, features: dict, mode: str, split: str) -> str:
        cum = features["rain_cum_may1"]
        verdict = "SAFE" if pred["class"] == 1 else "NOT SAFE"
        s = (f"The model classifies this date as {verdict} under the project's rainfall criterion "
             f"(raw model probability of 'safe': {pred['probability_safe']:.2f}, uncalibrated). "
             f"Reference check: rainfall since 1 May is {cum:.1f} mm, which "
             f"{'meets' if cum >= fu.CUM_MM else 'is below'} the {fu.CUM_MM} mm threshold of the rule. "
             "Whether the following 21 days bring enough rain without a long dry spell cannot be known on the date itself.")
        if mode == "historical_demo" and split == "train":
            s += " This year was part of the model's training data, so this demonstration is in-sample."
        elif mode == "historical_demo" and split == "held_out_test":
            s += " This year was held out from training."
        return s

    def need_data():
        if state["data"] is None:
            return _err(503, "dataset_unavailable", state["data_error"] or "historical dataset not loaded")
        return None

    def lookup(year: int, date_str: str):
        df = state["data"]
        m = df[(df.year == year) & (df.date_str == date_str)]
        return None if m.empty else m.iloc[0]

    # ------------------------------------------------------------------ pages / health
    @app.get("/")
    def index():
        return render_template("index.html", title="AI-Based Soybean Sowing Window Predictor")

    @app.get("/demo")
    def sowing_demo():
        # Presentation only: a static page whose numbers are produced by browser-side demo rules (static/js/demo_sim.js).
        # It does not call the model, the dataset or any /api/ endpoint.
        return render_template("demo.html", title="Sowing Prediction Demo (simulated data)")

    @app.get("/health")
    def health():
        ok = state["bundle"] is not None and state["data"] is not None
        body = {"status": "ok" if ok else "degraded", "model_loaded": state["bundle"] is not None,
                "dataset_loaded": state["data"] is not None, "live_weather_integration": False}
        errs = [e for e in (state["model_error"], state["data_error"]) if e]
        if errs:
            body["errors"] = errs
        if state["bundle"] and state["bundle"].warnings:
            body["warnings"] = state["bundle"].warnings
        return jsonify(body), (200 if ok else 503)

    @app.get("/api/model-info")
    def model_info():
        b = state["bundle"]
        if b is None:
            return _err(503, "model_unavailable", state["model_error"] or "model not loaded")
        m = b.metadata
        return jsonify({
            "title": m.get("title"), "location": m.get("location"), "target": m.get("target_definition"),
            "model": {k: (m.get("model") or {}).get(k) for k in ("label", "class", "params", "probability_calibrated", "decision_threshold")},
            "training_period": m.get("training_period"), "test_period": m.get("test_period"),
            "test_metrics": (m.get("metrics") or {}).get("test"),
            "baselines_test": (m.get("metrics") or {}).get("baselines_test"),
            "conclusions": m.get("conclusions"), "features": b.features,
            "feature_definitions": m.get("feature_definitions"),
            "supported_candidate_dates": m.get("supported_candidate_dates"),
            "limitations": LIMITATIONS, "warnings": b.warnings, "live_weather_integration": False})

    # ------------------------------------------------------------------ historical demo
    @app.get("/api/historical")
    def historical_index():
        bad = need_data()
        if bad:
            return bad
        years = [int(y) for y in sorted(state["data"].year.unique())]
        return jsonify({"years": years, "decision_dates": fu.DECISION_DATES,
                        "year_split": {str(y): split_of(y) for y in years},
                        "mode": "historical_demonstration",
                        "note": "Rows are archived ERA5 weather, not a current forecast."})

    @app.get("/api/historical/<int:year>/<date_str>")
    def historical_row(year: int, date_str: str):
        bad = need_data()
        if bad:
            return bad
        d, errs = fu.parse_candidate_date(date_str)
        if errs:
            return _err(422, "validation_error", "invalid date", errs)
        row = lookup(year, date_str) if d.year == year else None
        if row is None:
            return _err(404, "not_found", f"no historical record for year {year}, date {date_str}")
        feats = {c: _py(row[c]) for c in fu.FEATURE_COLUMNS}
        med = fu.reference_medians(state["data"], int(row["doy"]))
        return jsonify({"mode": "historical_demo", "candidate_date": date_str, "data_split": split_of(year),
                        "inputs": feats, "indicators": indicators(feats, med),
                        "retrospective_reference": retrospective(date_str, year)})

    # ------------------------------------------------------------------ prediction
    @app.post("/api/predict")
    def predict():
        if state["bundle"] is None:
            return _err(503, "model_unavailable", state["model_error"] or "model not loaded")
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return _err(400, "invalid_json", "request body must be a JSON object (Content-Type: application/json)")
        extra = sorted(set(body) - ALLOWED_PREDICT_KEYS)
        if extra:
            return _err(422, "validation_error", "unknown field(s) in request",
                        [{"field": k, "message": "unknown field"} for k in extra])
        mode = body.get("mode") or ("manual" if "features" in body else "historical")
        if mode not in ("historical", "manual"):
            return _err(422, "validation_error", "invalid mode",
                        [{"field": "mode", "message": "must be 'historical' or 'manual'"}])
        if "candidate_date" not in body:
            return _err(422, "validation_error", "candidate_date is required",
                        [{"field": "candidate_date", "message": "required (YYYY-MM-DD)"}])
        cdate, errs = fu.parse_candidate_date(body["candidate_date"])
        if errs:
            return _err(422, "validation_error", "invalid candidate_date", errs)
        date_str = cdate.isoformat()

        retro, split = None, "not_applicable"
        if mode == "historical":
            bad = need_data()
            if bad:
                return bad
            if "features" in body:
                return _err(422, "validation_error", "features must not be supplied in historical mode",
                            [{"field": "features", "message": "historical mode reads the stored dataset row"}])
            row = lookup(cdate.year, date_str)
            if row is None:
                return _err(404, "not_found",
                            f"no historical record for {date_str}; historical demo covers {fu.START_YEAR}-{fu.END_YEAR} "
                            f"on these dates: {', '.join(fu.DECISION_DATES)}")
            raw = {c: _py(row[c]) for c in fu.FEATURE_COLUMNS}
            feats, ferrs = fu.validate_feature_input(raw, cdate)
            if ferrs:
                return _err(503, "features_unavailable", "stored historical features failed validation", ferrs)
            retro, split, mode_name = retrospective(date_str, cdate.year), split_of(cdate.year), "historical_demo"
        else:
            if "features" not in body:
                return _err(422, "validation_error", "features are required in manual mode",
                            [{"field": "features", "message": "required"}])
            feats, ferrs = fu.validate_feature_input(body["features"], cdate)
            if ferrs:
                return _err(422, "validation_error", "invalid feature values", ferrs)
            mode_name = "manual_input"
        medians = fu.reference_medians(state["data"], int(feats["doy"])) if state["data"] is not None else None

        try:
            pred = inference.predict_one(state["bundle"], feats)
        except Exception:  # pragma: no cover - defensive
            app.logger.exception("prediction failed")
            return _err(500, "prediction_failed", "the model could not produce a prediction")

        meta = state["bundle"].metadata
        return jsonify({
            "prediction": {"label": pred["label"], "class": pred["class"],
                           "probability_safe": round(pred["probability_safe"], 4),
                           "probability_not_safe": round(pred["probability_not_safe"], 4),
                           "probability_note": "Raw classifier output; not calibrated and not a confidence score."},
            "candidate_date": date_str, "mode": mode_name, "data_split": split,
            "location": fu.LOCATION_NAME,
            "inputs": feats, "indicators": indicators(feats, medians),
            "reference_rule": {"description": f"rain since 1 May >= {fu.CUM_MM} mm (the observable part of the label rule)",
                               "met": bool(feats["rain_cum_may1"] >= fu.CUM_MM), "value_mm": feats["rain_cum_may1"]},
            "retrospective_reference": retro,
            "explanation": explanation(pred, feats, mode_name, split),
            "limitations": LIMITATIONS,
            "model": {"label": (meta.get("model") or {}).get("label"),
                      "trained_on_years": (meta.get("training_period") or {}).get("years"),
                      "dataset_fingerprint": (meta.get("dataset") or {}).get("fingerprint_sha256")},
        })

    # ------------------------------------------------------------------ errors
    @app.errorhandler(404)
    def not_found(_e):
        if request.path.startswith("/api/"):
            return _err(404, "not_found", "unknown API endpoint")
        return render_template("index.html", title="Not found"), 404

    @app.errorhandler(405)
    def bad_method(_e):
        return _err(405, "method_not_allowed", f"{request.method} is not allowed for {request.path}")

    @app.errorhandler(413)
    def too_large(_e):
        return _err(413, "payload_too_large", "request body too large")

    @app.errorhandler(500)
    def server_error(_e):
        return _err(500, "internal_error", "unexpected server error")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=os.environ.get("FLASK_DEBUG") == "1")
