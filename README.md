<div align="center">
 
# 🛡️ Phishing URL Detection using Ensemble Machine Learning

### Advanced ML Security System — Stacking & Voting Classifiers for Cybersecurity Threat Detection

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Domain](https://img.shields.io/badge/Domain-Cybersecurity%20%7C%20ML-critical?style=for-the-badge)]()

**By [Gayathri Chilukala](https://github.com/GayathriChilukala)**

</div>

---

## 📌 Project Overview

**Phishing URL detection** is a high-impact cybersecurity challenge — phishing attacks remain the leading vector for data breaches and financial fraud worldwide. This project builds a machine learning pipeline that analyzes the structural and lexical properties of URLs to classify them as **phishing** or **legitimate** with high accuracy.

What sets this project apart from basic classifiers is the deliberate comparison of **three advanced ensemble strategies** — Stacking, Hard Voting, and Soft Voting — to identify the most robust approach for real-world threat detection.

> **The Problem:** Phishing URLs are engineered to deceive. Simple rule-based detection fails against sophisticated attacks. Machine learning models that learn URL patterns offer a scalable, automated defense.
>
> **The Approach:** Extract 30+ structural URL features → compare ensemble methods → identify the best-performing model for deployment in a security pipeline.

---

## 🗂️ Repository Structure

```
Phishing_URL-S_Project/
├── Exploratory_data_analysis.ipynb   # Phase 1: EDA & dataset understanding
├── FeatureExtraction.ipynb            # Phase 2: URL feature engineering (30+ features)
├── StackingModel.ipynb                # Phase 3a: Stacking ensemble classifier
├── votingClassifierhard.ipynb         # Phase 3b: Hard voting ensemble
└── votingClassifiersoft.ipynb         # Phase 3c: Soft voting ensemble
```

Each notebook is a self-contained phase of the ML pipeline, structured to mirror a production-grade research workflow.

---

## 🔬 ML Pipeline — Phase by Phase

### Phase 1 · Exploratory Data Analysis
**`Exploratory_data_analysis.ipynb`**

A thorough investigation of the phishing URL dataset before any modeling — understanding data quality, class balance, and feature relationships.

- Dataset profiling: shape, data types, null value audit, class distribution
- Visual analysis: class imbalance plots, feature histograms, correlation heatmaps
- Statistical summary of URL characteristics across phishing vs. legitimate categories
- Identified key discriminative features that carry the most predictive signal
- Documented data quality issues and preprocessing decisions

---

### Phase 2 · Feature Extraction & Engineering
**`FeatureExtraction.ipynb`**

The most critical phase — transforming raw URL strings into a rich, machine-readable feature matrix. URL-based feature extraction requires domain expertise in both cybersecurity and data engineering.

**Feature Categories Extracted:**

| Category | Features |
|---|---|
| **URL Structure** | URL length, domain length, path depth, number of subdomains |
| **Special Characters** | Count of `@`, `-`, `_`, `//`, `?`, `=`, `&`, `%` symbols |
| **Lexical Patterns** | Presence of IP address, use of URL shortening services |
| **Domain Properties** | HTTPS usage, domain age indicators, prefix/suffix in domain |
| **Path Analysis** | Number of redirects, depth of directory path, file extension |
| **Suspicious Signals** | Presence of brand keywords, abnormal port usage, anchor tag ratio |

> **30+ features extracted** per URL, creating a structured feature matrix for ensemble model input.

---

### Phase 3a · Stacking Ensemble Classifier
**`StackingModel.ipynb`**

A **Stacking (Stacked Generalization)** ensemble that combines predictions from multiple base learners through a meta-learner, enabling the model to learn how to best aggregate individual classifier outputs.

**Architecture:**
```
Base Learners (Level 0)
  ├── Decision Tree Classifier
  ├── K-Nearest Neighbors
  ├── Support Vector Machine (SVM)
  └── Random Forest / Gradient Boosting

        ↓ out-of-fold predictions

Meta-Learner (Level 1)
  └── Logistic Regression
        ↓
   Final Prediction: Phishing / Legitimate
```

- K-fold cross-validation for out-of-fold base learner training (prevents data leakage)
- Meta-learner trained on stacked predictions for optimal combination
- Evaluated with Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

### Phase 3b · Hard Voting Classifier
**`votingClassifierhard.ipynb`**

A **Hard Voting** ensemble where each base classifier casts a vote and the majority class label wins. Prioritizes robustness and reliability over probability calibration.

- Combines multiple diverse classifiers into one unified predictor
- Each model contributes one vote — final class determined by majority
- Particularly effective when individual classifiers have complementary strengths
- Full evaluation with confusion matrix and classification report

---

### Phase 3c · Soft Voting Classifier
**`votingClassifiersoft.ipynb`**

A **Soft Voting** ensemble that averages the predicted **class probabilities** from each base classifier before making a final decision — yielding more nuanced, confidence-weighted predictions.

- Averages probability outputs across all base classifiers
- Outperforms hard voting when classifiers are well-calibrated
- Generates ROC curves and AUC scores for threshold analysis
- Enables confidence scoring for each URL prediction (useful in production pipelines)

---

## ⚔️ Model Comparison

| Ensemble Strategy | How It Works | Best When |
|---|---|---|
| **Stacking** | Meta-learner learns to combine base model outputs | Base models have different strengths; highest accuracy potential |
| **Hard Voting** | Majority vote across classifiers | Robustness matters; simple and interpretable |
| **Soft Voting** | Average of class probabilities | Classifiers are well-calibrated; richer confidence signals needed |

> The three approaches were implemented and compared to identify the optimal strategy for phishing detection — a methodology typical of **production ML research**.

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
|---|---|
| **Language** | Python 3.8+ |
| **ML Framework** | Scikit-learn |
| **Ensemble Methods** | StackingClassifier, VotingClassifier |
| **Base Models** | Decision Tree, KNN, SVM, Random Forest, Logistic Regression |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Feature Engineering** | Custom URL parsing with Python `re`, `urllib` |
| **Environment** | Jupyter Notebook |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Run the Pipeline

```bash
# 1. Clone the repository
git clone https://github.com/GayathriChilukala/Phishing_URL-S_Project.git
cd Phishing_URL-S_Project

# 2. Launch Jupyter
jupyter notebook

# 3. Run notebooks in order:
#    Step 1 → Exploratory_data_analysis.ipynb
#    Step 2 → FeatureExtraction.ipynb
#    Step 3 → StackingModel.ipynb (and/or voting classifiers)
```

> ⚠️ Run notebooks in sequence — `FeatureExtraction.ipynb` generates the feature matrix consumed by all three model notebooks.

---

## 💡 Why This Project Matters

Phishing is the most prevalent form of cybercrime globally, responsible for billions in annual losses. This project addresses a genuine security need with production-relevant techniques:

- **Scale:** Manual URL review doesn't scale to millions of daily web requests — automated ML detection is essential
- **Sophistication:** Ensemble methods are the go-to approach in competitive ML and real security systems (used by browser vendors, email filters, and endpoint security products)
- **Ensemble Comparison:** Systematically benchmarking Stacking vs. Hard Voting vs. Soft Voting mirrors the rigor expected in industry ML research and MLOps roles

---

## 🎯 Skills Demonstrated

This project showcases advanced skills relevant to **ML Engineer, Data Scientist, and Security ML** roles:

| Skill | Application |
|---|---|
| **Feature Engineering** | Custom URL parsing and 30+ feature extraction from raw strings |
| **Ensemble Learning** | Stacking, Hard Voting, Soft Voting — implemented and compared |
| **Model Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| **EDA** | Statistical profiling, class imbalance analysis, correlation analysis |
| **Cybersecurity Domain Knowledge** | URL structure, phishing indicators, web threat taxonomy |
| **Research Methodology** | Hypothesis-driven comparison of multiple model architectures |
| **Reproducible ML** | Modular notebook pipeline with clear phase separation |

---

## 📈 Potential Extensions

- Deploy the best-performing model as a **REST API** (Flask/FastAPI) for real-time URL scanning
- Integrate with a browser extension for **live phishing alerts**
- Add **WHOIS & DNS features** (domain age, registrar, TTL) for richer signals
- Train on a continuously updated dataset for **concept drift** resilience
- Apply **SHAP values** for feature importance explainability

---

## 👩‍💻 Author

**Gayathri Chilukala**

[![GitHub](https://img.shields.io/badge/GitHub-GayathriChilukala-181717?style=flat-square&logo=github)](https://github.com/GayathriChilukala)

---

<div align="center">

*Defending the web one URL at a time — with the power of ensemble machine learning.* 🛡️

</div>
