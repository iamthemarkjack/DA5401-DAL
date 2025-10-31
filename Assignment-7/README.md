# Multi-Class Model Selection using ROC and Precision-Recall Curves
**Assignment 7 | Rohith R - EP21B030**

---

## Overview
This assignment demonstrates comprehensive **multi-class model selection** using **ROC curves** and **Precision-Recall Curves (PRC)** on the UCI Landsat Satellite dataset. The analysis compares **six diverse classifiers** spanning different performance levels, including models expected to perform worse than random chance (AUC < 0.5), requiring careful interpretation of curves rather than relying solely on accuracy metrics.

The primary objective is to understand **threshold-independent evaluation** in multi-class settings using the **One-vs-Rest (OvR)** approach, and to identify the best-performing model through integrated analysis of ROC-AUC, Average Precision (AP), and traditional metrics.

---

## Dataset
**Landsat Satellite Dataset** (UCI Machine Learning Repository)
- **Classes**: 6 land cover types (excluding "all types present")
- **Features**: 36 spectral band values from satellite imagery
- **Samples**: 4,435 training samples, 2,000 test samples
- **Characteristics**: High-dimensional, moderate class overlap, non-trivial classification task

**Download**: [UCI ML Repository - Satimage Dataset](https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite)

---

## Models Evaluated

| Model | Library | Expected Performance |
|-------|---------|---------------------|
| K-Nearest Neighbors | `sklearn.neighbors.KNeighborsClassifier` | Moderate/Good |
| Decision Tree | `sklearn.tree.DecisionTreeClassifier` | Moderate |
| Dummy Classifier | `sklearn.dummy.DummyClassifier` | Baseline (AUC < 0.5) |
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | Good |
| Gaussian Naive Bayes | `sklearn.naive_bayes.GaussianNB` | Poor/Varies |
| Support Vector Machine | `sklearn.svm.SVC` | Good |

**Brownie Points**: Random Forest, XGBoost, and additional poor-performing models

---

## Usage
1. Download the Landsat dataset files (`sat.trn` and `sat.tst`) from the UCI ML Repository
2. Place the dataset files in the same directory as the notebook or update file paths accordingly
3. Open the Jupyter Notebook (`DA5401_A7_Landsat_Analysis.ipynb`)
4. Run all cells sequentially to reproduce the complete analysis

The notebook performs:
- **Part A**: Data loading, standardization, model training, and baseline evaluation (Accuracy, F1-score)
- **Part B**: Multi-class ROC curve generation using One-vs-Rest approach, macro-averaged AUC calculation, and identification of models with AUC < 0.5
- **Part C**: Precision-Recall curve analysis, Average Precision calculation, and interpretation of curve behavior for poor models
- **Part D**: Comprehensive ranking comparison across F1-score, ROC-AUC, and PRC-AP metrics with final model recommendation
- **Brownie Points**: Additional experiments with RandomForest, XGBoost, and other classifiers

---

## Key Results

### Baseline Performance
| Model | Accuracy | Weighted F1-Score |
|-------|----------|-------------------|
| K-Nearest Neighbors | 90.45% | 0.904 |
| SVM | 89.55% | 0.892 |
| Decision Tree | 85.05% | 0.851 |
| Logistic Regression | 83.95% | 0.830 |
| Gaussian Naive Bayes | 79.65% | 0.804 |
| Dummy Classifier | 23.05% | 0.086 |

### Key Findings
- **Best Overall Model**: Identified through comprehensive ROC-AUC and PRC-AP analysis
- **Worst Performer**: Dummy Classifier demonstrates AUC < 0.5 for minority classes
- **Trade-off Analysis**: ROC-AUC and PRC-AP rankings reveal precision-recall trade-offs
- **Threshold Robustness**: Top models maintain strong performance across decision thresholds

---

## Visualization Outputs
- Macro-averaged ROC curves for all 6 models
- Macro-averaged Precision-Recall curves for all 6 models
- Per-class PRC breakdown for worst-performing model
- Comparative bar charts across all evaluation metrics
- Ranking consistency analysis
