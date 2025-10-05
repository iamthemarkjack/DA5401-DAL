# Visualizing Data Veracity Challenges in Multi-Label Classification  
**Assignment 5 | Rohith R - EP21B030**  

---

## Overview  
This assignment explores **dimensionality reduction and visualization** on the Yeast multi-label dataset.  
It uses **t-SNE** and **Isomap** to project high-dimensional gene expression data into 2D space, helping to understand local clusters, global structure, and the challenges posed by overlapping or multi-label samples.

---

## Usage  
1. Open the Jupyter Notebook (`notebook.ipynb`).  
2. Place the Yeast dataset files (`yeast-train.arff` and `yeast-test.arff`) in the same directory as the notebook.  
3. Run all cells sequentially to reproduce data preprocessing, feature scaling, dimensionality reduction, visualizations, and comparative analysis of t-SNE and Isomap.  

The notebook performs:  
- Loading and exploration of the dataset  
- Feature scaling using standardization  
- Visualization of high-dimensional data in 2D  
- Comparison of local vs global structure and implications for classification  

---

## Requirements  
Install the required Python packages with:

```bash
pip install -r requirements.txt

