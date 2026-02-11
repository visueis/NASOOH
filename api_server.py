from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import os

# Simple API server for NASOOH:
# - Serves nasooh.html at /
# - Returns predictions from predictions_next_term.csv at /api/prediction?student_id=...

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "predictions_next_term.csv")
HTML_FILE = "nasooh.html"

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app)

@app.get("/")
def index():
    # Serve the main HTML file from the same folder
    return send_from_directory(BASE_DIR, HTML_FILE)

@app.get("/<path:filename>")
def static_files(filename):
    # Serve any other local assets (images/css/js) if referenced by nasooh.html
    return send_from_directory(BASE_DIR, filename)

def load_predictions():
    if not os.path.exists(CSV_PATH):
        return None
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df

@app.get("/api/prediction")
def prediction():
    student_id = (request.args.get("student_id") or "").strip()
    if not student_id:
        return jsonify({"error": "student_id required"}), 400

    df = load_predictions()
    if df is None:
        return jsonify({"error": "predictions_next_term.csv not found in the same folder as api_server.py"}), 500

    if "student_id" not in df.columns:
        return jsonify({"error": "CSV missing column: student_id"}), 500

    df["student_id"] = df["student_id"].astype(str)
    row = df[df["student_id"] == student_id]
    if row.empty:
        return jsonify({"error": "student_id not found"}), 404

    r = row.iloc[-1].to_dict()

    # Be flexible with column names
    student_name = r.get("student_name_en") or r.get("student_label") or r.get("student_name") or ""
    current_gpa = r.get("cum_gpa") or r.get("current_gpa") or r.get("gpa")
    predicted_gpa = r.get("predicted_next_term_gpa") or r.get("predicted_gpa") or r.get("pred")
    risk_level = r.get("predicted_risk_level") or r.get("risk_level") or r.get("risk")
    prob_high = r.get("risk_probability_high") or r.get("prob_high") or r.get("risk_prob")

    return jsonify({
        "student_id": student_id,
        "student_name": student_name,
        "current_gpa": current_gpa,
        "predicted_next_term_gpa": predicted_gpa,
        "risk_level": risk_level,
        "risk_probability_high": prob_high
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
