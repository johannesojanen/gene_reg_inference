# Gene Regulatory Network Inference from Time-Series Data Using a Dynamic Bayesian Network

This project implements a **Dynamic Bayesian Network (DBN)** to infer the structure of a 5-gene transcriptional regulatory network using time-course gene expression data from Cantone et al. (2009).

## Key Features
- **DBN Model:** Infers temporal dependencies using linear regression.
- **Performance Evaluation:** Includes ROC curve, AUC, and threshold analysis for interaction detection.
- **Data:** Uses the `switch-off experiment` dataset with 5 transcription factors measured over 190 minutes.

## Repository Structure
- `data`: Input time-course data and true network structure.
- `models`: DBN implementation.
- `evaluate_model`: Performance metrics and thresholding code.
- `test.ipynb`: Running the code for inference and evaluation.
- `report.pdf`: Detailed project report.