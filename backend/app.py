"""Flask API for the Customer Churn Prediction project.

Run from the project root:
    python backend/app.py
The API starts on http://localhost:5000

Endpoints
    GET  /api/health         -> is the API up and is the model loaded?
    GET  /api/options        -> allowed values, ranges, threshold and business assumptions (used to build the form)
    POST /api/predict        -> ONE customer as JSON -> churn probability, decision and recommended action
    POST /api/predict-batch  -> {"customers": [ ...up to 100... ]} -> one result per row (errors reported per row)

The model is trained in notebook/Customer_Churn_Prediction.ipynb and loaded ONCE when the server starts.
The decision threshold is read from models/model_info.json. We never retrain on a request.
"""
import json
import math
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.joblib")
INFO_PATH = os.path.join(BASE_DIR, "models", "model_info.json")

# Which website may call this API from a browser (the Vite dev server by default).
ALLOWED_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")
MAX_BATCH = 100

app = Flask(__name__)

# ---- load the model once at start-up ----
if not os.path.exists(MODEL_PATH) or not os.path.exists(INFO_PATH):
    raise SystemExit("Model files not found. Run notebook/Customer_Churn_Prediction.ipynb first to create models/.")
model = joblib.load(MODEL_PATH)
with open(INFO_PATH) as f:
    info = json.load(f)

NUMERIC = info["numeric_features"]
CATEGORICAL = info["categorical_features"]
INTEGER_FIELDS = set(info["integer_features"])
OPTIONAL_FIELDS = {"avg_data_gb"}            # may be missing: the pipeline fills it with the training median
THRESHOLD = info["threshold"]


@app.after_request
def add_cors_headers(response):
    """Allow the React app (a different port = a different 'origin') to call this API."""
    response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def is_blank(value):
    return value is None or (isinstance(value, str) and value.strip() == "")


def validate(payload):
    """Check one customer. Return (clean_row, errors); errors is a dict field -> message a person can understand."""
    if not isinstance(payload, dict):
        return None, {"body": "Send a JSON object with the customer details."}
    errors, row = {}, {}

    for field in NUMERIC:
        low, high = info["numeric_ranges"][field]
        value = payload.get(field)
        if is_blank(value):
            if field in OPTIONAL_FIELDS:
                row[field] = float("nan")        # allowed: filled by the pipeline
            elif field == "total_charges":
                row[field] = None                # decided below, once we know the tenure
            else:
                errors[field] = "This field is required."
            continue
        if isinstance(value, bool):              # JSON true/false is not a number here
            errors[field] = "Must be a number."
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors[field] = "Must be a number."
            continue
        if math.isnan(number) or math.isinf(number):
            errors[field] = "Must be a real number."
        elif field in INTEGER_FIELDS and not number.is_integer():
            errors[field] = "Must be a whole number."
        elif not (low <= number <= high):
            errors[field] = f"Must be between {low:g} and {high:g} (the range the model was trained on)."
        else:
            row[field] = number

    # Same rule as in the notebook: a blank total_charges is only valid for a brand-new customer (tenure 0 -> 0 rupees).
    if "total_charges" in row and row["total_charges"] is None:
        if row.get("tenure_months") == 0:
            row["total_charges"] = 0.0
        else:
            errors["total_charges"] = "Required (it may only be blank for a new customer with tenure 0)."
    if "senior_citizen" in row and row["senior_citizen"] not in (0, 1):
        errors["senior_citizen"] = "Must be 0 (no) or 1 (yes)."

    for field in CATEGORICAL:
        allowed = info["categories"][field]
        value = payload.get(field)
        if is_blank(value):
            errors[field] = "This field is required."
        elif str(value).strip() not in allowed:
            errors[field] = f"Unknown value '{value}'. Choose one of: {', '.join(allowed)}."
        else:
            row[field] = str(value).strip()

    # A simple consistency rule: tech support is only possible with internet
    if not errors and (row["internet_service"] == "No internet") != (row["tech_support"] == "No internet service"):
        errors["tech_support"] = "Use 'No internet service' exactly when internet_service is 'No internet'."
    return row, errors


def recommended_action(probability):
    """Translate a probability into plain advice for the retention team."""
    if probability >= max(0.7, THRESHOLD + 0.25):
        return "High risk: call the customer personally this week and offer the retention discount."
    if probability >= THRESHOLD:
        return "At risk: send the retention discount offer."
    if probability >= THRESHOLD / 2:
        return "Watch list: no discount yet; send a satisfaction survey and check open complaints."
    return "Low risk: no action needed."


def predict_rows(rows):
    """Predict for a list of already-validated rows (one model call for all of them)."""
    features = pd.DataFrame(rows, columns=NUMERIC + CATEGORICAL)
    probabilities = model.predict_proba(features)[:, 1]
    results = []
    for p in probabilities:
        p = float(p)
        results.append({
            "churn_probability": round(p, 4),
            "will_churn": p >= THRESHOLD,
            "decision": "Contact" if p >= THRESHOLD else "Do not contact",
            "recommended_action": recommended_action(p),
        })
    return results


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model": info["model_name"], "threshold": THRESHOLD})


@app.get("/api/options")
def options():
    return jsonify({
        "categories": info["categories"],
        "numeric_ranges": info["numeric_ranges"],
        "optional_fields": sorted(OPTIONAL_FIELDS),
        "threshold": THRESHOLD,
        "business_assumptions": info["business_assumptions"],
        "model": info["model_name"],
        "test_metrics": info["test_metrics"],
        "max_batch": MAX_BATCH,
    })


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":           # browser "preflight" check before a POST
        return ("", 204)
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON (Content-Type: application/json)."}), 400

    row, errors = validate(payload)
    if errors:
        return jsonify({"error": "Some inputs are invalid.", "fields": errors}), 422

    result = predict_rows([row])[0]
    result.update({
        "threshold": THRESHOLD,
        "model": info["model_name"],
        "note": "The probability is a risk score from the model. The decision uses the business threshold "
                f"{THRESHOLD:.2f} chosen in the notebook.",
    })
    return jsonify(result)


@app.route("/api/predict-batch", methods=["POST", "OPTIONS"])
def predict_batch():
    if request.method == "OPTIONS":
        return ("", 204)
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON (Content-Type: application/json)."}), 400

    # Accept {"customers": [...]} or a plain list [...]
    customers = payload.get("customers") if isinstance(payload, dict) else payload
    if not isinstance(customers, list):
        return jsonify({"error": "Send {\"customers\": [ ... ]} with a list of customer objects."}), 422
    if len(customers) == 0:
        return jsonify({"error": "The customer list is empty."}), 422
    if len(customers) > MAX_BATCH:
        return jsonify({"error": f"Too many customers: {len(customers)}. The limit is {MAX_BATCH} per request."}), 422

    # Validate every row; keep the good ones for ONE model call, remember errors per row.
    results = [None] * len(customers)
    good_rows, good_positions = [], []
    for i, customer in enumerate(customers):
        row, errors = validate(customer)
        customer_id = customer.get("customer_id") if isinstance(customer, dict) else None
        if errors:
            results[i] = {"row": i + 1, "customer_id": customer_id, "ok": False, "fields": errors}
        else:
            good_rows.append(row)
            good_positions.append((i, customer_id))

    if good_rows:
        for (i, customer_id), prediction in zip(good_positions, predict_rows(good_rows)):
            results[i] = {"row": i + 1, "customer_id": customer_id, "ok": True, **prediction}

    contact = sum(1 for r in results if r["ok"] and r["will_churn"])
    return jsonify({
        "results": results,
        "summary": {"total": len(results), "valid": len(good_rows), "invalid": len(results) - len(good_rows),
                    "to_contact": contact},
        "threshold": THRESHOLD,
        "model": info["model_name"],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
