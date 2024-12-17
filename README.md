# Inference of Transcriptional Regulatory Networks from Time-Series Data Using Dynamic Bayesian Networks


This project implements a **Dynamic Bayesian Network (DBN)** to infer the structure of a 5-gene transcriptional regulatory network using time-course gene expression data from Cantone et al. (2009).

## Reproduce Results

To reproduce the results, run the `run_model.ipynb` notebook. This will load the data, train the DBN model, and evaluate the performance of the model.

## Key Modules
- **DBN Model:** Infers temporal dependencies using linear regression.
- **Performance Evaluation:** Includes ROC curve, AUC, and threshold analysis for interaction detection.
- **Data:** Uses the `switch-off experiment` dataset with 5 transcription factors measured over 190 minutes.

## Repository Structure
- `run_model.ipynb`: Running the code for inference and evaluation.
- `data`: Input time-course data and true network structure.
- `models`: DBN implementation.
- `evaluate_model`: Performance metrics and thresholding code.
- `report.pdf`: Detailed project report.