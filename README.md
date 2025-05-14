# 🚀 MLflow Churn Prediction Starter Kit  
*A production-ready ML project template for learning MLflow*

![MLflow Logo](https://mlflow.org/images/MLflow-logo-final-whiteBG.png)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

This project demonstrates how to use **MLflow** to track and manage the end-to-end machine learning lifecycle for a **churn prediction** task using the Telco dataset.

Churn prediction is a classification problem where we identify which customers are likely to stop using a service. The model is trained on customer behavioral data and logs key metrics, parameters, and artifacts using MLflow.

---

## 🎯 Key Learning Objectives

- ✅ End-to-end ML workflow with MLflow tracking  
- ✅ Model packaging & reproducibility  
- ✅ Experiment tracking and comparison  
- ✅ Feature importance analysis  
- ✅ Conda environment management

---

## 🛠 Tech Stack

| Component      | Tool               |
|----------------|--------------------|
| Language       | Python 3.8         |
| ML Framework   | Scikit-learn       |
| MLOps Tool     | MLflow             |
| Environment    | Conda              |
| Algorithm      | RandomForest       |
| Data Handling  | pandas             |
| Visualization  | matplotlib         |
| Tracking UI    | MLflow UI on `localhost:5000` |

---

## ⚙️ Project Structure

```text
.
├── train.py                # Model training with MLflow logging
├── conda.yaml              # Conda environment definition
├── MLproject               # MLflow project config
├── data.csv                # Telco churn dataset
├── feature_importance.png  # Output plot (logged artifact)
```

---

## 📦 Installation & Setup

Make sure you have **Anaconda** or **Miniconda** installed on your Windows system.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/mlflow-churn-prediction.git
cd mlflow-churn-prediction
```

### 2️⃣ Create and Activate Conda Environment

```bash
conda env create -f conda.yaml
conda activate churn_prediction
```

### 3️⃣ (Optional) Verify MLflow Installation

```bash
mlflow --version
```

---

## 🚀 Run the Project

### 🔌 Start MLflow Tracking Server (Local)

```bash
mlflow server   --backend-store-uri sqlite:///mlflow.db   --default-artifact-root ./mlruns   -h 0.0.0.0 -p 5000
```

> ℹ️ Visit **http://127.0.0.1:5000** in your browser to access the MLflow UI.

### 🧠 Train the Model

```bash
python train.py --data_path data.csv --test_size 0.4
```

All logs, metrics, and artifacts will now be visible in the MLflow UI.

---

## 📊 What Gets Tracked?

| Type        | Description                               |
|-------------|-------------------------------------------|
| 🔢 Param    | `n_estimators`, `test_size`               |
| 📈 Metric   | `accuracy`                                |
| 🧠 Model    | RandomForestClassifier (via sklearn)      |
| 🖼️ Artifact | `feature_importance.png` (bar chart)      |

---

## 🧱 MLflow Architecture Overview

```text
+------------------------------------------------------------+
|                  MLflow Core Architecture                  |
+------------------------------------------------------------+
|                                                            |
|  +-------------+    +--------------+    +---------------+  |
|  |  Tracking   |<-->|   Projects   |<-->|    Models     |  |
|  |   Server    |    | (Packaging)  |    |  Registry     |  |
|  +-------------+    +--------------+    +---------------+  |
|        ^                                       ^           |
|        |                                       |           |
|        v                                       v           |
|  +----------------+                  +-------------------+ |
|  | Experiments &  |                  | Model Deployment  | |
|  |     Runs       |                  |    (Serving)      | |
|  +----------------+                  +-------------------+ |
|        ^                                              ^    |
|        |                                              |    |
|        v                                              v    |
|  +----------------+                        +------------------+ |
|  | Artifact Store |                        |   Integrations    | |
|  | (S3, GCS, etc) |                        | (Databricks, etc) | |
|  +----------------+                        +------------------+ |
+------------------------------------------------------------+
```

---

## 💡 Future Improvements

- Add model registry and versioning support  
- Serve model via `mlflow models serve`  
- Add unit tests with pytest  
- Dockerize the project  
- Add a Jupyter notebook for exploratory analysis  

---

## 🤝 Contributing

Contributions and suggestions are welcome!  
Feel free to open an [issue](https://github.com/rupesh40/mlflow-churn-prediction/issues) or a pull request to improve this learning resource.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🌟 Star this repo if it helped you learn something!

Happy learning! ✨
