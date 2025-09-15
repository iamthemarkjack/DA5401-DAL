# GMM-Based Synthetic Sampling for Imbalanced Data  
**Assignment 4 | Rohith R - EP21B030**

## Overview  
This assignment explores the use of **Gaussian Mixture Models (GMMs)** for addressing class imbalance in fraud detection.  
The idea is to model the **minority (fraudulent) class** distribution with a GMM and then generate synthetic samples that better capture its **multi-modal nature**.  

We compare three training strategies on a Logistic Regression classifier:  
- **Baseline** (original imbalanced data)  
- **GMM Oversampling** (synthetic minority samples)  
- **GMM + Clustering-Based Undersampling (CBU)** (reducing majority + augmenting minority)  

The performance is evaluated using **Precision, Recall, and F1-score** for both majority and minority classes, with emphasis on the fraud class.

---

## Usage  
1. Open the Jupyter Notebook file (`notebook.ipynb`).  
2. Download the dataset from Kaggle: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).  
3. Place the dataset (`creditcard.csv`) in the same directory as the notebook.  
4. Run all notebook cells sequentially to reproduce the **analysis, visualizations, and results**.  

The notebook handles:  
- Data loading and exploration  
- GMM fitting and synthetic sampling  
- GMM + CBU hybrid balancing  
- Training and evaluation of Logistic Regression models  
- Performance comparison plots and final recommendations  

---

## Requirements  
Install the required Python packages with:  

```bash
pip install -r requirements.txt

