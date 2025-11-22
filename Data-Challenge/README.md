## DA5401 — 2025 Data Challenge

This repository contains my solution for the **DA5401 Data Challenge**, a course project focused on predicting evaluation metric scores for large-language-model responses.  
The task provides conversation snippets (system prompt, user prompt, model response) and asks participants to predict a numerical score between 0 and 10.  
A key difficulty is the **distribution shift**: training labels are heavily concentrated at high scores (9–10), while the hidden test labels behave very differently.

## Objective

The goal of this project is to train a Transformer-based regression model that remains well-calibrated under distribution shift.  
To achieve this, the model is trained with a combination of:

- **Smooth L1 regression loss**  
- **Optimal Transport (OT) regularization** toward a broad **Uniform[2,8]** target distribution

The uniform prior serves as a simple, assumption-free calibration anchor that encourages the model to explore a realistic mid-range of scores rather than inheriting the skewed training distribution.

## How to Run

1. Place the following files in the working directory:
   - `train_data.json`
   - `test_data.json`

2. Run the training + inference script:
   ```bash
   python train_uniform_ot.py
   ```

3. The script will produce:
   ```
   submission.csv
   ```
   which contains model predictions (id, score) and is ready for submission to the challenge leaderboard.

## Files

- `train_uniform_ot.py` — main training/inference pipeline  
- `submission.csv` — output predictions  
- `README.md` — this document  