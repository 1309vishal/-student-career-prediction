from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

CAREER_DOMAINS: dict[str, dict[str, Any]] = {
    "Data Science & AI": {
        "weights": {"math": 0.35, "science": 0.25, "english": 0.15, "attendance": 0.10, "tech": 0.10, "soft": 0.05},
        "interests": {"data", "ai", "ml", "analytics", "statistics", "research", "math", "python"},
        "required_tech": ["Python", "SQL", "Statistics & Probability", "Machine Learning", "Data Visualization"],
        "required_soft": ["Analytical Thinking", "Problem Solving", "Data Storytelling"],
        "roadmap": [
            "Master Python data stack (NumPy, Pandas, Scikit-Learn) and SQL querying.",
            "Build 2 end-to-end predictive analytics projects with real-world datasets.",
            "Practice statistical hypothesis testing and model evaluation metrics.",
        ],
    },
    "Software & Full-Stack Engineering": {
        "weights": {"math": 0.25, "science": 0.25, "english": 0.15, "attendance": 0.15, "tech": 0.15, "soft": 0.05},
        "interests": {"coding", "software", "web", "app", "programming", "development", "cloud", "systems"},
        "required_tech": ["Data Structures & Algorithms", "Python / Java / JS", "Databases (SQL/NoSQL)", "Git & Version Control", "REST APIs"],
        "required_soft": ["Logical Reasoning", "Debugging Mindset", "Team Collaboration"],
        "roadmap": [
            "Strengthen Data Structures, Algorithms, and object-oriented system design.",
            "Develop and deploy a full-stack web application with authentication and database persistence.",
            "Contribute to collaborative repositories using Git workflows and unit testing.",
        ],
    },
    "Business & Financial Analytics": {
        "weights": {"math": 0.30, "science": 0.10, "english": 0.25, "attendance": 0.15, "tech": 0.05, "soft": 0.15},
        "interests": {"business", "finance", "economics", "banking", "strategy", "management", "consulting", "commerce"},
        "required_tech": ["Excel & Financial Modeling", "SQL", "BI Dashboards (Power BI / Tableau)", "Quantitative Aptitude"],
        "required_soft": ["Communication", "Decision Making", "Stakeholder Presentation"],
        "roadmap": [
            "Build interactive KPI dashboards in Power BI or Tableau using business datasets.",
            "Practice quantitative aptitude, forecasting, and financial ratio analysis.",
            "Present structured case-study insights translating numbers into business actions.",
        ],
    },
    "UI/UX & Product Design": {
        "weights": {"math": 0.10, "science": 0.15, "english": 0.30, "attendance": 0.15, "tech": 0.10, "soft": 0.20},
        "interests": {"design", "ui", "ux", "creative", "product", "frontend", "art", "human", "media"},
        "required_tech": ["Figma & Prototyping", "Design Systems", "HTML/CSS Basics", "User Research & Wireframing"],
        "required_soft": ["Empathy", "Visual Storytelling", "Communication"],
        "roadmap": [
            "Create end-to-end user flows, wireframes, and interactive prototypes in Figma.",
            "Conduct usability testing sessions and document heuristic improvements.",
            "Build a responsive frontend portfolio showcasing 3 complete case studies.",
        ],
    },
    "Core Engineering & IoT Systems": {
        "weights": {"math": 0.30, "science": 0.35, "english": 0.10, "attendance": 0.15, "tech": 0.05, "soft": 0.05},
        "interests": {"hardware", "iot", "robotics", "electronics", "mechanical", "embedded", "physics", "engineering"},
        "required_tech": ["Embedded C / Python", "Circuit & Sensor Integration", "CAD / Simulation Tools", "Mathematics"],
        "required_soft": ["Systematic Troubleshooting", "Attention to Detail", "Teamwork"],
        "roadmap": [
            "Prototype microcontroller and sensor workflows with real-time telemetry logging.",
            "Apply mathematical modeling and simulation tools to core engineering problems.",
            "Document hardware-software integration benchmarks and safety tolerances.",
        ],
    },
    "Management, Operations & EdTech": {
        "weights": {"math": 0.15, "science": 0.15, "english": 0.30, "attendance": 0.20, "tech": 0.05, "soft": 0.15},
        "interests": {"leadership", "operations", "teaching", "hr", "marketing", "public", "social", "communication"},
        "required_tech": ["Project Management Tools", "Spreadsheet Analytics", "Digital Communication Suites"],
        "required_soft": ["Leadership", "Public Speaking", "Conflict Resolution", "Time Management"],
        "roadmap": [
            "Lead a cross-functional campus or capstone project with clear milestone tracking.",
            "Develop structured presentation, negotiation, and team coordination habits.",
            "Earn foundational credentials in Agile project delivery and operations.",
        ],
    },
}


def _find_col(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    norm = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in norm:
            return norm[cand.lower()]
    return None


def _scale_factor(max_value: float) -> float:
    """Return the multiplier that brings a score column onto a 0-100 scale."""
    if not np.isfinite(max_value):
        return 1.0
    if max_value <= 10.0:
        return 10.0
    if max_value <= 20.0:
        return 5.0
    return 1.0


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"r2": r2, "rmse": rmse, "mae": mae}


class StudentCareerPredictor:
    def __init__(
        self,
        preprocessor: ColumnTransformer | None = None,
        regressor: Any = None,
        kmeans: KMeans | None = None,
        cluster_scaler: StandardScaler | None = None,
        cluster_labels_map: dict[int, str] | None = None,
        feature_columns: list[str] | None = None,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.preprocessor = preprocessor
        self.regressor = regressor
        self.kmeans = kmeans
        self.cluster_scaler = cluster_scaler
        self.cluster_labels_map = cluster_labels_map or {}
        self.feature_columns = feature_columns or []
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.metadata = metadata or {}
        self._target_name = (metadata or {}).get("target_name", "score")

    @property
    def ready(self) -> bool:
        return self.preprocessor is not None and self.regressor is not None

    def _prepare_dataframe(self, raw_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Split a raw CSV into a feature frame and a target vector.

        The target is never kept among the features, so the model cannot read
        the answer from its own inputs (no target leakage).
        """
        df = raw_df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        math_col = _find_col(df.columns, ("math score", "math_score", "math"))
        read_col = _find_col(df.columns, ("reading score", "reading_score", "reading"))
        write_col = _find_col(df.columns, ("writing score", "writing_score", "writing"))
        target_col = _find_col(
            df.columns,
            ("final_score", "average_score", "overall_score", "performance_index", "exam_score", "g3"),
        )

        if math_col and read_col and write_col:
            # Kaggle StudentsPerformance.csv: predict the writing score from the
            # math score, reading score and background factors.
            m = pd.to_numeric(df[math_col], errors="coerce")
            r = pd.to_numeric(df[read_col], errors="coerce")
            w = pd.to_numeric(df[write_col], errors="coerce")
            factor = _scale_factor(max(m.max(skipna=True), r.max(skipna=True), w.max(skipna=True)))
            df = df.drop(columns=[math_col, read_col, write_col])
            df["math_score"] = m * factor
            df["reading_score"] = r * factor
            y = (w * factor).to_numpy()
            self._target_name = "writing_score"
        elif target_col:
            y_series = pd.to_numeric(df[target_col], errors="coerce")
            y = (y_series * _scale_factor(y_series.max(skipna=True))).to_numpy()
            df = df.drop(columns=[target_col])
            self._target_name = target_col
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                raise ValueError(
                    "Could not detect score columns. Upload a CSV with score columns "
                    "(e.g., math score, reading score, writing score, or final_score)."
                )
            target_auto = numeric_cols[-1]
            y = pd.to_numeric(df[target_auto], errors="coerce").to_numpy()
            df = df.drop(columns=[target_auto])
            self._target_name = target_auto

        valid_mask = ~np.isnan(y)
        df = df.loc[valid_mask].reset_index(drop=True)
        y = y[valid_mask]

        if len(y) < 30:
            raise ValueError("Dataset must contain at least 30 valid student records.")

        # Attendance is used only when the dataset really contains it. It is never
        # generated artificially, because a made-up column would not be real evidence.
        att_col = _find_col(df.columns, ("attendance", "attendance_rate", "attendance_percentage", "absences"))
        if att_col is not None:
            if att_col.lower() == "absences":
                absences = pd.to_numeric(df[att_col], errors="coerce").fillna(4)
                df["attendance_pct"] = np.clip(100.0 - absences * 1.5, 50.0, 100.0)
                df = df.drop(columns=[att_col])
            elif att_col != "attendance_pct":
                df["attendance_pct"] = pd.to_numeric(df[att_col], errors="coerce")
                df = df.drop(columns=[att_col])

        return df, y

    def train(self, raw_df: pd.DataFrame, model_path: str | Path | None = None) -> dict[str, Any]:
        if raw_df.empty:
            raise ValueError("The uploaded CSV file is empty.")

        X_df, y = self._prepare_dataframe(raw_df)
        numeric_features = X_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [c for c in X_df.columns if c not in numeric_features]

        if not numeric_features and not categorical_features:
            raise ValueError("No usable feature columns found in the dataset.")

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42
        )

        transformers = []
        if numeric_features:
            transformers.append(
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                    numeric_features,
                )
            )
        if categorical_features:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("ohe", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_features,
                )
            )

        preprocessor = ColumnTransformer(transformers=transformers)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)

        candidates = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=42),
            "Random Forest": RandomForestRegressor(
                n_estimators=150, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1
            ),
        }

        comparison: dict[str, dict[str, float]] = {}
        best_name = ""
        best_r2 = -1e9

        for name, model in candidates.items():
            model.fit(X_train_proc, y_train)
            val_preds = model.predict(X_val_proc)
            metrics = _regression_metrics(y_val, val_preds)
            comparison[name] = metrics
            if metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                best_name = name

        # Refit best model on train+val and evaluate on held-out test set
        final_preprocessor = ColumnTransformer(transformers=transformers)
        X_tv_proc = final_preprocessor.fit_transform(X_train_val)
        X_test_proc = final_preprocessor.transform(X_test)

        final_model = candidates[best_name]
        final_model.fit(X_tv_proc, y_train_val)
        test_preds = final_model.predict(X_test_proc)
        test_metrics = _regression_metrics(y_test, test_preds)

        # K-Means Clustering on academic score + attendance profile
        cluster_cols = [c for c in ["math_score", "reading_score", "attendance_pct"] if c in X_df.columns]
        if not cluster_cols:
            cluster_cols = numeric_features[:3]

        cluster_matrix = X_df[cluster_cols].fillna(X_df[cluster_cols].median()).to_numpy()
        cluster_matrix = np.column_stack([cluster_matrix, y])
        cluster_scaler = StandardScaler()
        cluster_scaled = cluster_scaler.fit_transform(cluster_matrix)

        n_clusters = 3 if len(X_df) >= 30 else 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(cluster_scaled)

        # Rank clusters by mean target score so cluster names are meaningful
        cluster_means = {
            cid: float(np.mean(y[cluster_ids == cid])) for cid in range(n_clusters)
        }
        sorted_cids = sorted(cluster_means, key=lambda c: cluster_means[c], reverse=True)
        tier_names = ["High Achievers (Honors Track)", "Steady Performers (Growth Track)", "At-Risk Learners (Support Track)"]
        cluster_labels_map = {
            cid: tier_names[idx] if idx < len(tier_names) else f"Tier {idx + 1}"
            for idx, cid in enumerate(sorted_cids)
        }
        cluster_distribution = {
            cluster_labels_map[cid]: int(np.sum(cluster_ids == cid)) for cid in sorted_cids
        }

        feature_defaults: dict[str, Any] = {}
        for col in numeric_features:
            med = X_df[col].median()
            feature_defaults[col] = float(med) if pd.notna(med) else 0.0
        for col in categorical_features:
            mode = X_df[col].mode(dropna=True)
            feature_defaults[col] = str(mode.iloc[0]) if len(mode) else "unknown"

        self.preprocessor = final_preprocessor
        self.regressor = final_model
        self.kmeans = kmeans
        self.cluster_scaler = cluster_scaler
        self.cluster_labels_map = cluster_labels_map
        self.feature_columns = list(X_df.columns)
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        self.metadata = {
            "model_name": best_name,
            "validation_r2": float(best_r2),
            "dataset_rows": int(len(y)),
            "mean_score": float(np.mean(y)),
            "target_name": self._target_name,
            "has_attendance": "attendance_pct" in X_df.columns,
            "mean_attendance": float(np.mean(X_df["attendance_pct"])) if "attendance_pct" in X_df.columns else None,
            "feature_defaults": feature_defaults,
            "validation_comparison": comparison,
            "test_metrics": test_metrics,
            "test_size": int(len(y_test)),
            "cluster_distribution": cluster_distribution,
            "cluster_cols": cluster_cols,
        }

        if model_path:
            self.save(model_path)
        return self.metadata

    def analyze_student(self, profile: dict[str, Any]) -> dict[str, Any]:
        math_score = float(profile.get("math_score", 70.0))
        reading_score = float(profile.get("reading_score", 70.0))
        science_score = float(profile.get("science_score", (math_score + reading_score) / 2.0))
        attendance = float(profile.get("attendance_pct", 85.0))
        study_hours = float(profile.get("study_hours", 10.0))
        tech_rating = float(profile.get("tech_rating", 7.0))  # 1-10
        soft_rating = float(profile.get("soft_rating", 7.0))  # 1-10

        raw_tech_skills = [s.strip() for s in str(profile.get("tech_skills", "")).split(",") if s.strip()]
        raw_soft_skills = [s.strip() for s in str(profile.get("soft_skills", "")).split(",") if s.strip()]
        interests_text = str(profile.get("interests", "")).lower()

        # Predict academic performance using trained ML model or calibrated fallback
        if self.ready:
            defaults = self.metadata.get("feature_defaults", {})
            row_dict: dict[str, Any] = {}
            for col in self.feature_columns:
                low = col.lower()
                if "math" in low:
                    row_dict[col] = math_score
                elif "read" in low:
                    row_dict[col] = reading_score
                elif "science" in low:
                    row_dict[col] = science_score
                elif "attend" in low:
                    row_dict[col] = attendance
                elif "study_hours" in low or "hours" in low:
                    row_dict[col] = study_hours
                elif "prep" in low:
                    row_dict[col] = profile.get("test_prep") or defaults.get(col, "none")
                else:
                    # Anything the form does not ask for falls back to the typical
                    # (median / most common) value seen during training.
                    row_dict[col] = defaults.get(col, np.nan)
            input_df = pd.DataFrame([row_dict])
            X_proc = self.preprocessor.transform(input_df)
            predicted_score = float(np.clip(self.regressor.predict(X_proc)[0], 0.0, 100.0))
            model_used = self.metadata.get("model_name", "Trained Regressor")
        else:
            predicted_score = float(
                np.clip(
                    0.34 * math_score
                    + 0.30 * reading_score
                    + 0.16 * science_score
                    + 0.12 * attendance
                    + min(study_hours * 0.8, 10.0),
                    0.0,
                    100.0,
                )
            )
            model_used = "Baseline Weighted Estimator (Train CSV for ML Model)"

        # Determine K-Means cluster tier
        if self.kmeans is not None and self.cluster_scaler is not None:
            cluster_cols = self.metadata.get("cluster_cols", [])
            vec_vals = []
            for c in cluster_cols:
                if "math" in c:
                    vec_vals.append(math_score)
                elif "read" in c:
                    vec_vals.append(reading_score)
                elif "attend" in c:
                    vec_vals.append(attendance)
                else:
                    vec_vals.append(predicted_score)
            vec_vals.append(predicted_score)
            scaled_pt = self.cluster_scaler.transform([vec_vals])
            cid = int(self.kmeans.predict(scaled_pt)[0])
            cluster_tier = self.cluster_labels_map.get(cid, "Steady Performers")
        else:
            if predicted_score >= 80 and attendance >= 85:
                cluster_tier = "High Achievers (Honors Track)"
            elif predicted_score >= 62 and attendance >= 75:
                cluster_tier = "Steady Performers (Growth Track)"
            else:
                cluster_tier = "At-Risk Learners (Support Track)"

        # Strengths & Weaknesses Diagnosis
        metrics_map = {
            "Mathematics & Quantitative": math_score,
            "Reading & Communication": reading_score,
            "Science & Analytical": science_score,
            "Attendance Consistency": attendance,
            "Technical Proficiency": tech_rating * 10.0,
            "Soft Skills & Collaboration": soft_rating * 10.0,
        }
        sorted_metrics = sorted(metrics_map.items(), key=lambda x: x[1], reverse=True)
        strengths = [{"area": k, "score": round(v, 1)} for k, v in sorted_metrics[:3] if v >= 65]
        weaknesses = [{"area": k, "score": round(v, 1)} for k, v in reversed(sorted_metrics) if v < 75][:3]
        if not strengths:
            strengths = [{"area": sorted_metrics[0][0], "score": round(sorted_metrics[0][1], 1)}]

        # Career Recommendation Engine
        interest_tokens = set(re.findall(r"[a-z]+", interests_text))
        tech_lower = {s.lower() for s in raw_tech_skills}
        soft_lower = {s.lower() for s in raw_soft_skills}

        career_matches = []
        for domain, cfg in CAREER_DOMAINS.items():
            w = cfg["weights"]
            base_fit = (
                w["math"] * math_score
                + w["science"] * science_score
                + w["english"] * reading_score
                + w["attendance"] * attendance
                + w["tech"] * (tech_rating * 10.0)
                + w["soft"] * (soft_rating * 10.0)
            )
            interest_hits = len(interest_tokens & cfg["interests"])
            interest_bonus = min(interest_hits * 4.5, 13.5)

            req_tech = cfg["required_tech"]
            req_soft = cfg["required_soft"]
            matched_tech = [
                rt for rt in req_tech if any(rt.lower() in ts or ts in rt.lower() for ts in tech_lower)
            ]
            matched_soft = [
                rs for rs in req_soft if any(rs.lower() in ss or ss in rs.lower() for ss in soft_lower)
            ]
            skill_bonus = (len(matched_tech) / max(len(req_tech), 1)) * 6.0 + (
                len(matched_soft) / max(len(req_soft), 1)
            ) * 4.0

            match_pct = round(min(base_fit * 0.82 + interest_bonus + skill_bonus, 99.0), 1)
            missing_tech = [rt for rt in req_tech if rt not in matched_tech]
            missing_soft = [rs for rs in req_soft if rs not in matched_soft]

            career_matches.append(
                {
                    "domain": domain,
                    "match_pct": match_pct,
                    "matched_tech": matched_tech,
                    "missing_tech": missing_tech,
                    "matched_soft": matched_soft,
                    "missing_soft": missing_soft,
                    "roadmap": cfg["roadmap"],
                }
            )

        career_matches.sort(key=lambda item: item["match_pct"], reverse=True)
        top_career = career_matches[0]

        # Attendance risk alert
        attendance_status = (
            "Optimal (>= 85%)"
            if attendance >= 85
            else "Moderate (75% - 84%)"
            if attendance >= 75
            else "Critical (< 75% — Attendance Intervention Needed)"
        )

        return {
            "student_name": profile.get("student_name") or "Student",
            "predicted_score": round(predicted_score, 1),
            "grade_band": (
                "A+ (Distinction)"
                if predicted_score >= 90
                else "A (First Class)"
                if predicted_score >= 80
                else "B (Above Average)"
                if predicted_score >= 70
                else "C (Average)"
                if predicted_score >= 60
                else "D (Needs Academic Support)"
            ),
            "model_used": model_used,
            "cluster_tier": cluster_tier,
            "attendance": round(attendance, 1),
            "attendance_status": attendance_status,
            "metrics_breakdown": {k: round(v, 1) for k, v in metrics_map.items()},
            "strengths": strengths,
            "weaknesses": weaknesses,
            "career_matches": career_matches[:4],
            "top_career": top_career,
            "raw_tech_skills": raw_tech_skills,
            "raw_soft_skills": raw_soft_skills,
        }

    def save(self, path: str | Path) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "preprocessor": self.preprocessor,
                "regressor": self.regressor,
                "kmeans": self.kmeans,
                "cluster_scaler": self.cluster_scaler,
                "cluster_labels_map": self.cluster_labels_map,
                "feature_columns": self.feature_columns,
                "numeric_features": self.numeric_features,
                "categorical_features": self.categorical_features,
                "metadata": self.metadata,
            },
            dest,
        )

    @classmethod
    def load(cls, path: str | Path) -> "StudentCareerPredictor":
        src = Path(path)
        if not src.exists():
            return cls()
        data = joblib.load(src)
        return cls(**data)
