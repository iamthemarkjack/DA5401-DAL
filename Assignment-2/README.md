# Dimensionality Reduction and Classification with PCA
Assignment - 2 | Rohith R - EP21B030  

## Project Overview  
This project explores the impact of PCA on classification performance for the **Mushroom dataset**.  
The workflow includes:  
- One-hot encoding of categorical features.  
- Standardization of features.  
- Dimensionality reduction with PCA and visualization of the transformed space.  
- Comparison of Logistic Regression performance on the **original dataset** vs the **PCA-transformed dataset**.  

The key objective is to assess whether PCA can simplify the dataset while retaining predictive power.  

## Key Findings  
- PCA successfully reduced the high-dimensional dataset to a much smaller set of components while retaining ~95% variance.  
- Visualization of the first few principal components shows **substantial separation** between edible and poisonous mushrooms, though with some overlap.  
- Logistic Regression on PCA-transformed data achieves **comparable accuracy** to the original dataset, with only a minor loss in precision/recall.  
- PCA improved interpretability and reduced redundancy, but did not yield a dramatic boost in performance since the original dataset was already highly separable.  

## Conclusion
PCA reduces feature redundancy and dimensionality while maintaining nearly the same classification performance. Logistic Regression serves as a good surrogate model to evaluate PCA’s usefulness, though more complex models could be more effective and may reveal additional benefits.  

## Usage  
1. Open the notebook file (`notebook.ipynb`).  
2. Run all cells in sequence to reproduce the analysis, visualizations, and results.  
3. The notebook is self-contained and manages preprocessing, PCA, and classification.  

## Requirements  
Install the required Python packages using:  

```bash
pip install -r requirements.txt
