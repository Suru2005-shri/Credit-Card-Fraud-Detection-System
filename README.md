# 💳 Credit Card Fraud Detection System

A Machine Learning project designed to detect fraudulent credit card transactions using imbalanced data handling techniques and classification models. This project simulates a real-world fraud detection pipeline used in banking and fintech systems.

---

## 📌 Overview

Credit card fraud detection is a critical problem in the financial industry due to the massive volume of transactions and the rarity of fraudulent cases.

This project builds a complete pipeline to:

* Analyze transaction data
* Handle class imbalance
* Train ML models
* Detect fraudulent transactions
* Simulate real-time fraud alerts

---

## 🎯 Problem Statement

Fraudulent transactions represent a very small fraction of total transactions, making detection difficult.

Challenges:

* Highly imbalanced dataset
* Risk of false positives (blocking genuine users)
* Risk of false negatives (missing fraud)

---

## 💡 Solution Approach

The project follows a structured Machine Learning pipeline:

1. Data preprocessing and cleaning
2. Feature scaling and transformation
3. Handling imbalance using SMOTE
4. Model training using classification algorithms
5. Evaluation using appropriate metrics
6. Fraud prediction and alert simulation

---

## 🛠 Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn
* **Imbalance Handling:** SMOTE (imbalanced-learn)
* **Models:** Logistic Regression, Random Forest
* **Visualization:** Matplotlib, Seaborn

---

## 🏗 Project Architecture

```
Transaction Data
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
SMOTE Balancing
      ↓
Model Training
      ↓
Evaluation
      ↓
Fraud Prediction
      ↓
Alert System
```

---

## 📂 Folder Structure

```
Credit-Card-Fraud-Detection/
│
├── data/          # Dataset files
├── notebooks/     # Jupyter notebooks (EDA)
├── src/           # Core scripts
├── models/        # Saved models
├── outputs/       # Results and reports
├── images/        # Visualizations
├── main.py        # Main execution file
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Credit-Card-Fraud-Detection-System.git

# Navigate to folder
cd Credit-Card-Fraud-Detection-System

# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python main.py
```

---

## 📊 Model Evaluation

Evaluation metrics used:

* Precision
* Recall
* F1-score
* Confusion Matrix

Why not accuracy?
Because fraud detection datasets are highly imbalanced, accuracy alone is misleading.

---

## 📈 Results

* Successfully detected fraudulent transactions
* Improved recall using SMOTE
* Balanced precision vs recall trade-off

---

## 🎭 Fraud Detection Simulation

The system simulates transaction behavior and predicts:

* ✅ Normal Transaction
* 🚨 Fraudulent Transaction

Example:

```
Transaction Amount: 2500
Prediction: FRAUD
```

---

## 📸 Outputs

The project generates:

* Confusion Matrix
* Data distribution graphs
* Fraud vs Normal comparison charts

---

## 🚀 Future Improvements

* Real-time API integration (Flask/FastAPI)
* Deployment using Streamlit dashboard
* Advanced models (XGBoost, Deep Learning)
* Feature importance and explainability (SHAP)

---

## 💼 Use Case

This project demonstrates:

* Handling real-world imbalanced datasets
* End-to-end ML pipeline development
* Practical fraud detection concepts used in banking

---

## 👨‍💻 Author
Sarthak Dhumal

Developed as part of a Data Science and Machine Learning portfolio project for internships and placements.

---

## ⭐ If you found this useful, consider giving it a star!
