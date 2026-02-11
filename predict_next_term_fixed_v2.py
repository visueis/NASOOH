#!/usr/bin/env python3
"""
Predict next term GPA and risk for each student's latest completed term.

Outputs a CSV with (student_id, student_name/label, latest term stats, predicted_next_term_gpa, predicted_risk_level, risk_probability_high)
"""
import argparse, json, os
import pandas as pd
import numpy as np
import joblib

def term_sort_key(term_key: str):
    # term_key examples: 1447-1, 1447-2, 1447-S
    if not isinstance(term_key, str) or "-" not in term_key:
        return (0, 0)
    year, term = term_key.split("-", 1)
    try:
        y = int(year)
    except Exception:
        y = 0
    if str(term).upper() == "S":
        t = 3
    else:
        try:
            t = int(term)
        except Exception:
            t = 0
    return (y, t)

def read_sheet_case_insensitive(xlsx_path: str, requested: str) -> pd.DataFrame:
    try:
        return pd.read_excel(xlsx_path, sheet_name=requested)
    except ValueError:
        available = pd.ExcelFile(xlsx_path).sheet_names
        lower_map = {s.lower(): s for s in available}
        key = requested.lower()
        if key in lower_map:
            return pd.read_excel(xlsx_path, sheet_name=lower_map[key])
        # fallbacks
        for cand in ["Term_Summary", "term_summary", "02_term_summary", "ML_Dataset", "ml_dataset"]:
            if cand.lower() in lower_map:
                return pd.read_excel(xlsx_path, sheet_name=lower_map[cand.lower()])
        raise ValueError(f"Worksheet '{requested}' not found. Available sheets: {available}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to transcripts_clean_en.xlsx")
    ap.add_argument("--sheet", default="Term_Summary", help="Sheet name (default: Term_Summary)")
    ap.add_argument("--models_dir", default="models", help="Directory containing saved models")
    ap.add_argument("--out", default="predictions_next_term.csv", help="Output CSV filename")
    args = ap.parse_args()

    reg_path = os.path.join(args.models_dir, "gpa_regressor.joblib")
    clf_path = os.path.join(args.models_dir, "risk_classifier.joblib")
    cfg_path = os.path.join(args.models_dir, "config.json")

    reg = joblib.load(reg_path)
    clf = joblib.load(clf_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    feature_cols = cfg.get("feature_cols", [])

    ts = read_sheet_case_insensitive(args.input, args.sheet)

    # Ensure numeric cols
    for col in ["term_gpa", "term_points", "cum_gpa", "term_registered_hours", "cum_points"]:
        if col in ts.columns:
            ts[col] = pd.to_numeric(ts[col], errors="coerce")

    # Keep completed terms only: term_gpa > 0 and term_points > 0
    if "term_gpa" not in ts.columns or "term_points" not in ts.columns:
        raise KeyError("Term_Summary must contain 'term_gpa' and 'term_points' columns.")
    completed = ts[(ts["term_gpa"] > 0) & (ts["term_points"] > 0)].copy()

    if completed.empty:
        raise ValueError("No completed terms found (term_gpa>0 and term_points>0).")

    # latest completed term per student
    if "student_id" not in completed.columns or "term_key" not in completed.columns:
        raise KeyError("Term_Summary must contain 'student_id' and 'term_key' columns.")
    completed["__key"] = completed["term_key"].apply(term_sort_key)
    latest = completed.sort_values(["student_id", "__key"]).groupby("student_id").tail(1)

    # Build feature matrix
    missing = [c for c in feature_cols if c not in latest.columns]
    if missing:
        raise KeyError(f"Missing required feature columns in Term_Summary: {missing}")
    X = latest[feature_cols].copy()

    pred_gpa = reg.predict(X)
    pred_risk = clf.predict(X)

    proba = None
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)
        classes = list(getattr(clf, "classes_", []))
        if "HIGH" in classes:
            proba = probs[:, classes.index("HIGH")]
        else:
            proba = probs.max(axis=1)

    # Determine display label column
    if "student_label" in latest.columns:
        label_col = "student_label"
    elif "student_name_en" in latest.columns:
        label_col = "student_name_en"
    elif "student_name" in latest.columns:
        label_col = "student_name"
    else:
        label_col = None

    base_cols = ["student_id", "term_key", "term_gpa", "cum_gpa"]
    if label_col:
        base_cols.insert(1, label_col)

    out = latest[base_cols].copy()
    if label_col and label_col != "student_label":
        out = out.rename(columns={label_col: "student_label"})
    elif not label_col:
        out["student_label"] = out["student_id"]

    out["predicted_next_term_gpa"] = pred_gpa
    out["predicted_risk_level"] = pred_risk
    if proba is not None:
        out["risk_probability_high"] = proba

    out = out.sort_values("student_id")
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("Wrote:", args.out)

if __name__ == "__main__":
    main()
