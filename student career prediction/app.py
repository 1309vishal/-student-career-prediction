from __future__ import annotations

import os
import secrets
from pathlib import Path

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

from predictor import StudentCareerPredictor

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "student_career_model.joblib"
ALLOWED_DATASET_EXTENSIONS = {".csv"}
MAX_UPLOAD_BYTES = 64 * 1024 * 1024

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

predictor = StudentCareerPredictor.load(MODEL_PATH)


@app.get("/")
def index():
    return render_template("index.html", model=predictor.metadata, analysis=None, form={})


@app.post("/analyze")
def analyze():
    try:
        profile = {
            "student_name": request.form.get("student_name", "").strip() or "Student",
            "math_score": float(request.form.get("math_score", 75)),
            "reading_score": float(request.form.get("reading_score", 75)),
            "science_score": float(request.form.get("science_score", 75)),
            "attendance_pct": float(request.form.get("attendance_pct", 88)),
            "study_hours": float(request.form.get("study_hours", 12)),
            "test_prep": request.form.get("test_prep", "completed"),
            "tech_rating": float(request.form.get("tech_rating", 7)),
            "soft_rating": float(request.form.get("soft_rating", 7)),
            "tech_skills": request.form.get("tech_skills", ""),
            "soft_skills": request.form.get("soft_skills", ""),
            "interests": request.form.get("interests", ""),
        }
        for key in ("math_score", "reading_score", "science_score", "attendance_pct"):
            if not (0.0 <= profile[key] <= 100.0):
                raise ValueError(f"{key.replace('_', ' ').title()} must be between 0 and 100.")

        analysis = predictor.analyze_student(profile)
        return render_template(
            "index.html",
            model=predictor.metadata,
            analysis=analysis,
            form=request.form,
        )
    except ValueError as exc:
        flash(str(exc), "error")
    except Exception:
        app.logger.exception("Unable to analyze student profile")
        flash("Analysis failed unexpectedly. Check your inputs and try again.", "error")
    return redirect(url_for("index"))


@app.post("/train")
def train():
    upload = request.files.get("dataset")
    if not upload or not upload.filename:
        flash("Choose the Kaggle Student Performance CSV before training.", "error")
        return redirect(url_for("index"))
    if Path(upload.filename).suffix.lower() not in ALLOWED_DATASET_EXTENSIONS:
        flash("Training dataset must be a .csv file.", "error")
        return redirect(url_for("index"))

    try:
        dataset = pd.read_csv(upload, low_memory=False)
        predictor.train(dataset, MODEL_PATH)
        flash(
            f"Training complete. Selected {predictor.metadata['model_name']} "
            f"(Validation R2: {predictor.metadata['validation_r2']:.3f}).",
            "success",
        )
    except ValueError as exc:
        flash(str(exc), "error")
    except Exception:
        app.logger.exception("Unable to train student performance model")
        flash("Training failed unexpectedly. Check the CSV columns and try again.", "error")
    return redirect(url_for("index"))


@app.errorhandler(413)
def request_too_large(_error):
    flash("The uploaded file exceeds the 64 MB limit.", "error")
    return render_template("index.html", model=predictor.metadata, analysis=None, form={}), 413


if __name__ == "__main__":
    app.run(debug=False)
