# Imputation via Regression for Missing Data 
**Assignment 6 | Rohith R - EP21B030** ---

## Overview  
This assignment explores and compares different **missing data handling strategies** on a classification dataset, specifically focusing on **listwise deletion** and various **single imputation methods**. The imputation techniques include **mean/median**, **linear regression**, and **non-linear regression**, with the performance of each strategy evaluated using key classification metrics (Accuracy, Precision, Recall, F1-score).

---

## Usage  
1. Open the Jupyter Notebook (`notebook.ipynb`) or review the provided performance analysis (tables and plots).  
2. Ensure the relevant dataset (e.g., UCI Credit Card Default Clients Dataset) is loaded and preprocessed for missing data simulation or analysis.  
3. The notebook is self-contained run all cells sequentially to reproduce the missing data handling steps, model training, performance metric calculations, and comparative visualizations.  

The notebook performs:  
- Loading and initial exploration of the dataset.  
- Implementation of four missing data strategies: Listwise Deletion (D), Median Imputation (A), Linear Regression Imputation (B), and KNN Regressor Imputation (C).  
- Training of a classifier (e.g., Logistic Regression or similar) on each processed dataset.  
- Quantitative comparison of models using Accuracy, Precision, Recall, and F1-score.  
- Visualization of performance using bar plots and radar charts.  

---

## Requirements  
Install the required Python packages with:

```bash
pip install -r requirements.txt