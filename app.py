# -*- coding: utf-8 -*-
"""
app.py
======
Flask web server for the Credit Card Fraud Detection Dashboard.

Routes:
  GET  /           -> Serve the dashboard (dashboard/index.html)
  POST /predict    -> Run ML prediction on a single transaction
  GET  /health     -> Health check

Usage:
    python app.py

Author  : Fraud Detection System
Version : 1.0
"""

import sys
import json
import numpy as np
from pathlib import Path
from flask import Flask, send_from_directory, request, jsonify

# ── Make src/ importable ───────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.predict import predict_single  # noqa: E402

# ── App setup ──────────────────────────────────────────────────────────────
DASHBOARD_DIR = BASE_DIR / "dashboard"

app = Flask(__name__, static_folder=str(DASHBOARD_DIR), static_url_path="")


# ==========================================================================
# Routes
# ==========================================================================

@app.route("/")
def index():
    """Serve the main dashboard page."""
    return send_from_directory(str(DASHBOARD_DIR), "index.html")


@app.route("/health")
def health():
    """Simple health check endpoint."""
    return jsonify({"status": "ok", "service": "Credit Card Fraud Detection API"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict fraud probability for a single transaction.

    Expects JSON body:
    {
        "Amount": float,
        "Time"  : float,         (optional, defaults to 43200 = noon)
        "V1"..  "V28": float,    (optional, default 0.0)
        "threshold": float       (optional, default 0.5)
    }

    Returns JSON:
    {
        "prediction" : "FRAUD" | "LEGIT",
        "probability": float (0-1),
        "risk_level" : str,
        "alert"      : str
    }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({"error": "Invalid or missing JSON body"}), 400

        # ── Build transaction dict ─────────────────────────────────────────
        threshold = float(data.pop("threshold", 0.5))

        # Ensure required keys exist with sensible defaults
        transaction = {
            "Time"  : float(data.get("Time",   43200)),   # default = noon
            "Amount": float(data.get("Amount", 0.0)),
        }
        # V1 .. V28  (default to 0.0 if not provided)
        for i in range(1, 29):
            key = f"V{i}"
            transaction[key] = float(data.get(key, 0.0))

        # ── Run prediction ─────────────────────────────────────────────────
        result = predict_single(transaction, threshold=threshold)
        return jsonify(result)

    except FileNotFoundError:
        return jsonify({
            "error": "Model artifacts not found. Please run the training pipeline first: python main.py"
        }), 503

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ==========================================================================
# Entry point
# ==========================================================================
if __name__ == "__main__":
    print("""
+--------------------------------------------------------------+
|   [CC]  CREDIT CARD FRAUD DETECTION SYSTEM  v1.0            |
|   Flask Dashboard Server                                     |
+--------------------------------------------------------------+
  Dashboard : http://127.0.0.1:5000
  Predict   : POST http://127.0.0.1:5000/predict
  Health    : http://127.0.0.1:5000/health
+--------------------------------------------------------------+
""")
    app.run(host="0.0.0.0", port=5000, debug=False)
