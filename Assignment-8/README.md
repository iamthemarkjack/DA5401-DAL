# Ensemble Learning for Complex Regression Modeling on Bike Share Data  
**Assignment 8 | Rohith R – EP21B030**

---

## Overview
This assignment explores **ensemble learning techniques** — **Bagging**, **Boosting**, and **Stacking** — to solve a complex regression problem predicting **hourly bike rental demand**.  
The goal is to analyze how ensemble methods reduce **bias** and **variance**, and how combining diverse learners through stacking can yield superior performance.

The task involves building models on the **UCI Bike Sharing Demand Dataset** and evaluating them using the **Root Mean Squared Error (RMSE)** metric.

---

## Dataset
**Bike Sharing Demand Dataset** (UCI Machine Learning Repository)  
- **Samples** : 17,379 hourly records  
- **Target Variable** : `cnt` – total count of rented bikes per hour  
- **Features** : Weather conditions, time, and seasonality factors  
- **Characteristics** : Non-linear relationships, temporal dependencies, and high variance  

**Download:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)

---

## Models Implemented

| Model | Type | Purpose |
|-------|------|----------|
| Decision Tree Regressor | Single Model | Non-linear baseline |
| Linear Regression | Single Model | Linear baseline |
| Bagging Regressor | Ensemble | Variance reduction |
| Gradient Boosting Regressor | Ensemble | Bias reduction |
| Stacking Regressor | Meta-Ensemble | Optimal bias–variance trade-off |

---

## Usage
1. Download and extract the **Bike Sharing Dataset** (`hour.csv`) into the working directory.  
2. Open the notebook:  
   `notebook.ipynb`  
3. Run all cells sequentially to reproduce:
   - Data preprocessing and one-hot encoding  
   - Baseline model training (Decision Tree / Linear Regression)  
   - Ensemble methods: Bagging → Boosting → Stacking  
   - RMSE comparison and bias–variance discussion  

The notebook outputs clear RMSE tables, visual comparisons, and explanatory markdown for bias–variance effects.

---

## Key Results

### RMSE Comparison
| Model | RMSE (lower = better) | Observation |
|:------|:----------------------|:-------------|
| Baseline (Decision Tree / Linear) | ~ X.XXX | Starting point |
| Bagging Regressor | ↓ | Reduced variance |
| Gradient Boosting Regressor | ↓↓ | Reduced bias |
| Stacking Regressor | **↓↓↓ (best)** | Combines diversity for optimal trade-off |

*(Actual RMSE values depend on random seed and data splits.)*

---

## Conceptual Insights

### Bagging
Averaging multiple models trained on bootstrap samples reduces variance:  
\[
\text{Var}[\hat{f}_{\text{bag}}] = \rho\sigma^2 + \frac{(1-\rho)}{M}\sigma^2
\]

### Boosting
Sequentially corrects residual errors to reduce bias:  
\[
\hat{f}_M(x) = \sum_{m=1}^M \gamma_m h_m(x)
\]

### Stacking
Learns an optimal combination of heterogeneous base models through a meta-learner:  
\[
\hat{f}_{\text{stack}}(x) = g(h_1(x), h_2(x), \dots, h_k(x))
\]

Stacking typically outperforms both Bagging and Boosting by learning how to weight model strengths adaptively.

---

## Visual Outputs
- RMSE bar plot comparing all models  
- Correlation heatmap of base learner predictions  
- Predicted vs Actual scatter plots for top models  
- Markdown explanations of bias–variance trade-off and stacking intuition  

---

## Summary
- **Bagging** → reduces **variance**  
- **Boosting** → reduces **bias**  
- **Stacking** → balances both using **model diversity + meta-learning**

> **Best Model:** Stacking Regressor  
> **Reason:** Combines diverse models to minimize both bias and variance, achieving the lowest RMSE.

---
