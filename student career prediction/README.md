# EduPath — Student Academic Performance & Career Prediction

A local Flask analytics application that compares supervised regression models and applies unsupervised clustering on the Kaggle Student Performance dataset to predict academic scores, segment student archetypes, analyze skill gaps, and recommend tailored career domains and learning roadmaps.

## Requirements

- Python 3.10 or newer
- The Kaggle **Student Performance** CSV (e.g., `StudentsPerformance.csv` or `student-mat.csv`), available from [Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- A small synthetic file, `sample_data/students_performance_sample.csv`, is included only so you can try the app quickly. Use the real Kaggle CSV for your final results.

## Run locally

Recommended with `uv`:

```powershell
uv venv .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Or create the virtual environment with your installed Python:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:5000>, upload the Kaggle Student Performance CSV in the model panel, and select **Train with dataset**. The model predicts the **writing score** from the math score, reading score and background factors (the target is never used as an input). Training compares **Linear Regression**, **Decision Tree Regressor**, and **Random Forest Regressor** on a validation split, evaluates the winning model on a held-out test split, and fits **K-Means Clustering** to segment student performance tiers. The fitted pipeline is saved locally under `artifacts/`.

## Features

- **Academic Performance Prediction**: Predicts final academic score / GPA percentage using marks, attendance, study hours, and preparation factors.
- **Model Comparison & Clustering**: Compares Linear Regression, Decision Tree, and Random Forest ($R^2$, RMSE, MAE) alongside K-Means student clustering.
- **Strengths & Weaknesses Diagnosis**: Breaks down subject-level scores, attendance consistency, and technical vs. soft skills.
- **Career Recommendation Engine**: Matches academic profile, technical skills, soft skills, and interests against core career domains (Data Science & AI, Software Engineering, Core Engineering & Systems, Business & Product Analytics, UI/UX & Digital Design, Management & Consulting).
- **Skill Gap & Learning Roadmap**: Identifies missing competencies for the top career match and generates prioritized learning recommendations.
- **Interactive Analytics Dashboard**: Built-in visual charts for model $R^2$ comparison, cluster distribution, subject score breakdown, and skill gap progress (pure HTML/CSS bars) with zero external CDN dependencies.

## Notes on data

- Attendance is used for training only if the CSV really has an attendance (or absences) column. It is never generated artificially.
- In the analysis form, attendance, skills and interests are entered by the user and feed the diagnosis and career engine.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests check that the target is not leaked into the features, that training and saving work, and that the web routes respond correctly.
