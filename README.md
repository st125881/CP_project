# Predicting Total Battery Cycle Life Using Machine Learning

**Final Project – AST02.10 Computer Programming for Data Science and Artificial Intelligence**  
Asian Institute of Technology (AIT)

**Instructor:** Dr. Chantri Polprasert

**Authors:**
- Sabrina Afrin (st127171)
- Somkamon Mettawiharee (st125881)

---

## 1. Project Overview

This project aims to predict the **total cycle life of lithium-ion batteries using only their first 50 cycles**.

The project uses the Materials Aging and Reliability Dataset (MATR), containing 134 LFP/graphite battery cells and 114,313 cycle records.

Six features are extracted from the first 50 cycles of each battery and used to train and evaluate several machine learning regression models.

The models include:
- Linear Regression (Baseline)
- Random Forest
- Tuned Random Forest
- XGBoost
- Tuned XGBoost

The best-performing model, **Tuned Random Forest**, is deployed as a web application using Streamlit.

### Live Application

The application is deployed on Streamlit Community Cloud.

**[Open Battery Cycle Life Prediction App](https://cpproject-hxzybmgh3ywwnxpmrbhau6.streamlit.app/)**

No installation is required to use the online application.

---

## 2. Model Performance

The models were evaluated using 27 unseen test batteries.

| Model | MAE (cycles) | RMSE (cycles) | R² |
|---|---:|---:|---:|
| Mean Predictor | 292.41 | 395.01 | N/A |
| Linear Regression | 199.17 | 268.43 | 0.5107 |
| Random Forest | 161.82 | 253.70 | 0.5629 |
| **Tuned Random Forest** | **151.24** | **240.78** | **0.6063** |
| XGBoost | 168.45 | 272.42 | 0.4960 |
| Tuned XGBoost | 171.04 | 259.21 | 0.5437 |

The Tuned Random Forest achieved the lowest prediction error and was selected as the final model for deployment.

---

## 3. Repository Structure

```text
CP_project/
│
├── models/
│   ├── model_meta.json
│   └── tuned_rf.joblib
│
├── .dockerignore
├── app.py
├── Dockerfile
├── features.py
├── README.md
├── requirements.txt
├── sample_battery_50_cycles.csv
└── train.py
```

### Main Files

| File | Description |
|---|---|
| `app.py` | Streamlit web application |
| `features.py` | Feature engineering shared by training and application |
| `train.py` | Model training script |
| `requirements.txt` | Required Python packages |
| `models/tuned_rf.joblib` | Saved Tuned Random Forest model |
| `models/model_meta.json` | Model information, metrics, and feature ranges |
| `sample_battery_50_cycles.csv` | Sample battery data for testing |

---

## 4. Requirements

To run the application locally, you need:

- Python 3.10 or newer
- pip
- Git

## 5. Installation and Setup

**Step 1: Clone the repository**

```bash
git clone https://github.com/st125881/CP_project.git
cd CP_project
```

**Step 2: Create a virtual environment**

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

For macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Step 3: Install the required packages**

```bash
pip install -r requirements.txt
```

---

## 6. Run the Streamlit Application

After installing the dependencies, start the application using:

```bash
streamlit run app.py
```

Open your browser and go to:

```text
http://localhost:8501
```

The application provides three main functions:

- **Upload Cycle Data:** Upload a CSV file containing at least 50 cycles from one battery. The application extracts the required features and predicts the total cycle life.
- **Enter Values:** Manually enter the six feature values (mean values over the first 50 cycles) to obtain a prediction.
- **About Model:** View model information, evaluation results, and training feature ranges.

A sample CSV file (`sample_battery_50_cycles.csv`) is included in the main project folder for testing the application.

---
## 7. Deploy on Streamlit Community Cloud (Free)

The application is deployed using Streamlit Community Cloud.

To deploy your own version of this application:

1. Push the project repository to GitHub. Make sure the `models/` folder, including `tuned_rf.joblib` and `model_meta.json`, is included.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and sign in using your GitHub account.
3. Click **Create app**, select your GitHub repository, and set the main file path to `app.py`.
4. Click **Deploy**. After a few minutes, your application will be available through a public `*.streamlit.app` link.

**Live Application:** [Battery Cycle Life Prediction App](https://cpproject-hxzybmgh3ywwnxpmrbhau6.streamlit.app/)
---

## 8. Features Used

The model uses six selected features, calculated as the mean values over the first 50 cycles of each battery.

| Feature | Description |
|---|---|
| `energy_difference` | Difference between charge energy and discharge energy |
| `time_temperature_integrated` | Integrated time-temperature measure |
| `dc_internal_resistance` | DC internal resistance |
| `charge_energy` | Energy recorded during charging |
| `temperature_minimum` | Minimum temperature in each cycle |
| `temperature_range` | Difference between maximum and minimum temperature |

---

## 9. Limitations

- The model was trained using only one battery type (A123 APR18650M1A LFP/graphite).
- The evaluation was conducted using a single train-test split of 107 training batteries and 27 test batteries.
- Predictions are intended for early battery screening, not exact cycle-life estimation.
- The model may be less accurate for batteries with characteristics outside the training data.
- Feature selection was performed before the train-test split, which may result in slightly optimistic evaluation scores.

---

## 10. Dataset and References

**Dataset:** Materials Aging and Reliability Dataset (MATR)

https://data.matr.io/1/projects/5c48dd2bc625d700019f3204

**Reference:**

Severson, K. A., Attia, P. M., Jin, N., et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy, 4*, 383–391.

https://doi.org/10.1038/s41560-019-0356-8

