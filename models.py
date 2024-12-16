import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


class DynamicBayesianNetworkModel:
    """
    A simple first-order Dynamic Bayesian Network model using linear regressions.
    For each gene G_i, we fit a regression: G_i(t) ~ all genes(t-1).
    We store p-values or coefficient magnitudes in a 5x5 score matrix.
    Rows = target gene at time t, Cols = regulator gene at time (t-1).
    """

    def __init__(self, genes, scoring="neglog_pval"):
        """
        Args:
            genes (list): List of gene names, e.g. ['CBF1', 'GAL4', 'SWI5', 'GAL80', 'ASH1'].
            scoring (str): The metric to fill the 5x5 matrix with. Options:
                - "neglog_pval": Use -log10(p-value) as the score (larger => more significant).
                - "coef": Use the raw coefficient magnitude (absolute value).
        """
        self.genes = genes
        self.score_matrix = None
        self.scoring = scoring

    def fit(self, time_data: pd.DataFrame):
        """
        Fit the DBN model: for each gene, regress G_i(t) on all genes at time t-1.

        Args:
            time_data (pd.DataFrame): Must have columns for each gene in self.genes
                                      and rows for consecutive time points.
        """
        data_subset = time_data[self.genes].copy()

        # We'll shift the data by 1 to create (t, t-1) pairs
        # For time series of length T, we'll get T-1 pairs
        data_t = data_subset.iloc[1:].reset_index(drop=True)  # G(t), from time 1..T-1
        data_t_lag = data_subset.iloc[:-1].reset_index(drop=True)  # G(t-1), from time 0..T-2

        n_genes = len(self.genes)
        scores = np.zeros((n_genes, n_genes))

        for i, target_gene in enumerate(self.genes):
            # target = G_i(t)
            y = data_t[target_gene].values

            # predictors = all genes(t-1)
            X = data_t_lag[self.genes].values

            # We'll use statsmodels to get p-values for each coefficient
            # statsmodels requires adding a constant intercept manually
            X_sm = sm.add_constant(X)
            model = sm.OLS(y, X_sm)
            results = model.fit()

            # Retrieve p-values for each gene(t-1) predictor
            # results.pvalues is length n_genes+1 (the first is for the intercept).
            pvals = results.pvalues[1:]  # skip intercept
            coefs = results.params[1:]

            for j in range(n_genes):
                if self.scoring == "neglog_pval":
                    # Use -log10(p-value); handle pval=0 with a small epsilon
                    pval = pvals[j]
                    if pval <= 1e-300:
                        score = 300.0  # cap at some large number
                    else:
                        score = -np.log10(pval)
                elif self.scoring == "coef":
                    score = abs(coefs[j])
                else:
                    raise ValueError("Unsupported scoring method.")

                # row i = target_gene, col j = predictor gene
                scores[i, j] = score

        score_df = pd.DataFrame(scores, index=self.genes, columns=self.genes)
        # set diagonal to NaN so we don't consider self-self edges
        np.fill_diagonal(score_df.values, 0)
        self.score_matrix = score_df

    def get_scores(self):
        """
        Return the 5x5 score matrix.
        Rows = target gene at time t, Cols = gene at time t-1.
        """
        return self.score_matrix