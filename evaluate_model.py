import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import auc
import numpy as np
import pandas as pd


def evaluate_performance(inferred_structure: pd.DataFrame, true_structure: pd.DataFrame) -> dict:
    # Ensure the matrices have the same shape and order
    if not inferred_structure.index.equals(true_structure.index) or not inferred_structure.columns.equals(
            true_structure.columns):
        raise ValueError("Inferred and true structures must have the same genes in the same order.")

    # Create a mask to exclude diagonal entries
    mask = ~np.eye(inferred_structure.shape[0], dtype=bool)

    # Flatten the structures into 1D arrays, excluding diagonals
    true_flat = true_structure.values[mask]
    inferred_flat = inferred_structure.values[mask]

    # Convert to binary
    true_binary = (true_flat != 0).astype(int)
    inferred_binary = (inferred_flat != 0).astype(int)

    # Calculate True Positives (TP) and False Positives (FP)
    true_positives = np.sum((inferred_binary == 1) & (true_binary == 1))
    false_positives = np.sum((inferred_binary == 1) & (true_binary == 0))

    # Calculate totals
    total_true_interactions = np.sum(true_binary)
    total_possible_non_interactions = len(true_binary) - total_true_interactions

    # Compute fractions
    true_positive_fraction = true_positives / total_true_interactions if total_true_interactions > 0 else 0
    false_positive_fraction = false_positives / total_possible_non_interactions if total_possible_non_interactions > 0 else 0

    # Build and return the metrics dictionary
    metrics = {
        'true_positive_fraction': true_positive_fraction,
        'false_positive_fraction': false_positive_fraction,
        'true_positive_count': true_positives,
        'false_positive_count': false_positives,
        'total_true_interactions': total_true_interactions,
        'total_possible_non_interactions': total_possible_non_interactions,
    }

    return metrics


def evaluate_thresholds(true_structure: pd.DataFrame, inferred_structures: dict) -> tuple:
    roc_data = []
    for thr, inf_struct in inferred_structures.items():
        # Evaluate performance with the updated function that masks diagonals
        m = evaluate_performance(inf_struct, true_structure)
        roc_data.append((thr, m['true_positive_fraction'], m['false_positive_fraction'],
                         m['true_positive_count'], m['false_positive_count']))
    roc_df = pd.DataFrame(roc_data, columns=['threshold', 'tpr', 'fpr', 'tp_count', 'fp_count']).sort_values(
        'threshold')

    # Compute area under ROC curve
    auc_value = auc(roc_df['fpr'], roc_df['tpr'])

    # Find all thresholds that ensure all interactions are found (tpr=1)
    full_detection = roc_df[roc_df['tpr'] == 1.0]
    if not full_detection.empty:
        # Find the minimum and maximum thresholds among those that achieve full detection
        min_threshold = full_detection['threshold'].min()
        max_threshold = full_detection['threshold'].max()

        # Find the threshold with the minimum number of false positives
        best_row = full_detection.loc[full_detection['fp_count'].idxmin()]
        threshold_for_full_detection = best_row['threshold']
        false_positives_at_that_threshold = best_row['fp_count']
    else:
        threshold_for_full_detection = None
        false_positives_at_that_threshold = None
        min_threshold = None
        max_threshold = None

    return roc_df, auc_value, {
        'best_threshold': threshold_for_full_detection,
        'false_positives': false_positives_at_that_threshold,
        'min_threshold': min_threshold,
        'max_threshold': max_threshold
    }


def evaluate_model(model, true_structure, above=True, below=False):
    # Display the true structure
    print("True Structure:")
    display(true_structure)

    # Get the model's score matrix and display it
    score_matrix = model.get_scores()
    print("Inferred Score Matrix:")
    display(score_matrix)

    # Prepare thresholds
    thr_start = score_matrix.values.min()
    thr_stop = score_matrix.values.max()
    thresholds = np.linspace(thr_start, thr_stop, 100)

    # Build a dictionary of inferred structures at each threshold
    inferred_structures = {}

    # Use thresholds (hashable floats) as keys
    if above:
        for thr in thresholds:
            binary_matrix = (score_matrix >= thr).astype(int)
            inferred_structures[thr] = binary_matrix

    if below:
        for thr in thresholds:
            binary_matrix = (score_matrix <= thr).astype(int)
            inferred_structures[thr] = binary_matrix

    # Evaluate performance across thresholds
    roc_df, auc_value, full_detection_info = evaluate_thresholds(true_structure, inferred_structures)

    # Plot ROC curve

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Optional: Set a seaborn style for better aesthetics
    # sns.set(style='whitegrid')

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(
        roc_df['fpr'],
        roc_df['tpr'],
        color='darkorange',
        lw=2,
        label=f'ROC curve (AUC = {auc_value:.2f})'
    )

    # Plot the diagonal line for a random classifier
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Midline')

    # Customize the plot
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # Add gridlines
    plt.grid(True, linestyle='--', alpha=0.5)

    # Enhance tick parameters
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Print thresholds related to full detection
    best_threshold = full_detection_info['best_threshold']
    false_positives_at_best = full_detection_info['false_positives']
    min_threshold = full_detection_info['min_threshold']
    max_threshold = full_detection_info['max_threshold']

    if best_threshold is not None:
        print(f"Best Threshold for Full Detection: {best_threshold}")
        print(f"False Positives at Best Threshold: {false_positives_at_best}")
        print(f"Minimum Threshold for Full Detection: {min_threshold}")
        print(f"Maximum Threshold for Full Detection: {max_threshold}")

        # Display the best inferred structure
        best_structure = inferred_structures[best_threshold].copy()

        # Create a mask to exclude diagonal entries
        mask = ~np.eye(best_structure.shape[0], dtype=bool)
        best_structure_no_diag = best_structure.where(mask, 0)

        print("Best Inferred Structure (Excluding Diagonals):")
        display(best_structure_no_diag)
    else:
        print("No threshold produced full detection of interactions")