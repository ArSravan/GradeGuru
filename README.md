# GradingGuru
Academic performance &amp; risk prediction with fairness analysis

# GradeGuru — Student Academic Performance & Risk Predictor  
Flask · Machine Learning · Docker

---

##  Project Overview

*GradeGuru* is a web-based machine learning application that predicts a student’s *final academic grade (G3)* and classifies their *academic risk level* (*High / Medium / Low*) using demographic, behavioral, and academic features.

The project is designed as a *full-stack Python web application* using *Flask, with a trained **ML regression model, a clean **frontend interface, a **REST API, and full **Docker containerization*.

This project was developed as part of the *Python module (Semester 1)* in the *MSc Data Science & Artificial Intelligence* program at *SRH University*.

---

## Team & Collaboration

*Team Members*
- *Sravan*  
  - Machine Learning model training  
  - API design & validation  
  - Docker configuration  
  - Backend integration & debugging  

- *Rizwan*  
  - Frontend UI (HTML/CSS/JS)  
  - Page routing (Home / Dashboard / Result)  
  - UX improvements  
  - End-to-end testing  

Collaboration was handled via *GitHub*, with both members contributing commits, testing across different machines, and resolving integration issues together.

---

## Dataset

- *UCI Student Performance Dataset*
- Contains student demographic, social, and academic features
- Target variable: *G3 (Final Grade)*

The dataset is used to train *regression models* that estimate a student’s final grade and derive a risk category.

---

## Application Features

### Web Application (Frontend)
- *Home Page (/)*  
  Introduction and navigation entry point

- *Dashboard (/dashboard)*  
  Interactive form to input student features  
  Supports:
  - *Early Mode* (no prior grades)
  - *Full Mode* (includes G1 & G2)

- *Result Page (/result)*  
  Displays:
  - Predicted final grade (G3)
  - Risk band (High / Medium / Low)

---

### REST API (Backend)
- GET /api/health  
  Health check endpoint

- GET /api/schema  
  Returns required input features for:
  - Early mode
  - Full mode

- POST /api/predict  
  Accepts JSON input and returns:
  - Predicted G3
  - Risk band

---

### Risk Band Logic
| Predicted G3 | Risk Level |
|-------------|-----------|
| < 10      | High      |
| 10–13.99  | Medium    |
| ≥ 14      | Low       |

---

## Project Structure

```text
GradeGuru/
├── Dockerfile
├── requirements.txt
├── wsgi.py                # Flask entrypoint (create_app)
├── app/
│   ├── __init__.py
│   └── api.py             # REST API blueprint
├── models/
│   ├── early.joblib
│   ├── full.joblib
│   └── metadata.json
├── scripts/
│   └── train_models.py    # Model training script
└── templates/
    ├── index.html
    ├── dashboard.html
    └── result.html

---
```
## Requirements

Before running the project, make sure you have:

* Python **3.10 – 3.12**
* Git
* Docker (optional, only if you want to run the container)

Check versions:

bash
python --version
git --version
docker --version


*Windows (PowerShell)*

powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1


---

### 3) Install dependencies

Ensure the virtual environment is activated before running this.

bash
pip install --upgrade pip
pip install -r requirements.txt


If this step fails, stop and fix the error. Re-running will not magically solve version conflicts.

---

### 4) Run the application locally

This project provides a wsgi.py, which is the primary entry point.

bash
python wsgi.py


If the above command fails, fall back to Flask CLI.

*macOS / Linux*

bash
export FLASK_APP=app
export FLASK_ENV=development
flask run


*Windows (PowerShell)*

powershell
$env:FLASK_APP="app"
$env:FLASK_ENV="development"
flask run


If Flask cannot locate the application, update FLASK_APP to the module that defines Flask(__name__), for example:

* app:app
* app.main:app
* app.app:app

---

### 5) Access the application

Once the server starts successfully, open a browser and navigate to:

text
http://127.0.0.1:5000


If nothing loads, the application did not start correctly. Check the terminal output.

---

### 6) Run using Docker (optional)

Use this if you want environment isolation or deployment parity.

#### Build the Docker image

bash
docker build -t gradeguru .


#### Run the container

bash
docker run --rm -p 5000:5000 gradeguru


Then open:

text
http://127.0.0.1:5000


If the port does not work, inspect the EXPOSE directive in the Dockerfile.

---

### 7) Model training (required if predictions fail)

If the application throws errors related to missing model files, run the training scripts.

bash
python scripts/train.py
python scripts/evaluate.py


After successful execution, confirm that trained artifacts exist in:

text
models/


If model files are missing, the application will not be able to perform inference.

---

### 8) Common issues

* *ModuleNotFoundError*

  * Run commands from the project root
  * Ensure the virtual environment is active

* *Model file not found*

  * Training was not executed
  * File paths or filenames do not match what the app expects

* *Port already in use*

bash
flask run --port 5001


or

bash
docker run --rm -p 5001:5000 gradeguru


* *Dependency installation failure*

  * Read the error message
  * Fix the specific version or system dependency issue
  * Do not retry blindly
