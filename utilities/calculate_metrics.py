# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 23:03:48 2024

@author: Leila Jafari Khouzani
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn import metrics
from itertools import cycle
from sklearn.preprocessing import label_binarize, OneHotEncoder

from utilities.config import cfg

# =============================================================================
# function 1:       calculate_metrics
# =============================================================================
def calculate_classification_metrics(clf, clf_name,
                                     X, y_true,
                                     plot_roc=False,
                                     y_pred_return=False,
                                     label='',
                                     return_dict=False):
    """
    Computes per-class and overall metrics, with optional ChemBLE-specific processing.
    """
    # Initialize variables
    y_pred_proba = None
    loss = None

    if hasattr(clf, 'predict_proba'):
        y_pred_proba = clf.predict_proba(X)

        # Apply zero bias regulator for non-ChemBLE case
        y_pred_proba[:, 1] *= cfg.zero_bias_regulator

        y_pred_indices = np.argmax(y_pred_proba, axis=1)
        y_pred = np.array([cfg.dataset.class_labels[i] for i in y_pred_indices])
        # loss = metrics.log_loss(y_true, y_pred_proba)
        loss = loss_categorical_cross_entropy(y_true, y_pred_proba, one_hot=False)
    else:
        y_pred = clf.predict(X)
        # Calculate all metrics using the separate function
    metrics_dict = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        average=cfg.metrics.average,
        loss=loss,
        plot_roc=plot_roc,
        plot_suffix=clf_name,
        plot_label=label
    )

    if return_dict:
        return metrics_dict

    prepared_row = prepare_metrics_for_saving(metrics_dict)
    if y_pred_return:
        return y_pred, prepared_row
    return prepared_row


# =============================================================================
# function 2:       calculate_metrics
# =============================================================================
def calculate_metrics(y_true, y_pred,
                      y_pred_proba=None,
                      average='weighed',
                      loss=None,
                      plot_roc=False,
                      plot_suffix='',
                      plot_label=''):
    """
    Calculate classification metrics and optionally plot ROC/PR curves.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for ROC AUC)        
        average: Averaging method for metrics
        loss: Precomputed loss value
        plot_roc: Whether to plot ROC and PR curves
        plot_suffix: Suffix for plot titles (typically classifier name)
        plot_label: Additional label for plots
        
    Returns:
        Dictionary containing overall and per-class metrics
    """
    # Calculate overall metrics
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average=average, zero_division=0)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)

    # Calculate per-class metrics
    precision_per_class = metrics.precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = metrics.recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = metrics.f1_score(y_true, y_pred, average=None, zero_division=0)

    # Calculate ROC AUC if probabilities are available
    roc_auc = np.nan
    if y_pred_proba is not None:
        onehot_encoder = OneHotEncoder(sparse_output=False, categories='auto')
        one_hot_labels = onehot_encoder.fit_transform(y_true.reshape(-1, 1))
        roc_auc = metrics.roc_auc_score(one_hot_labels, y_pred_proba,
                                        multi_class="ovr", average=average)

        # Plot curves if requested
        if plot_roc and y_pred_proba is not None:
            plot_roc_curve(y_true, y_pred_proba, plot_suffix, plot_label=plot_label)
            plot_pr_curves(y_true, y_pred_proba, cfg.dataset.class_labels,
                           plot_suffix, plot_label=plot_label)
    # Log confusion matrix
    confusion_mat(y_true, y_pred, suffix=plot_suffix, cmdir=plot_label)

    return {
        'overall': {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc,
            'roc_auc': roc_auc
        },
        'per_class': {
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'class_labels': cfg.dataset.class_labels
        }
    }


# =============================================================================
# function 3:
# =============================================================================
def prepare_metrics_for_saving(metrics_dict):
    #overall metrics
    row = {}
    for k, v in metrics_dict['overall'].items():
        row[f'overall_{k}'] = v

    # per-class metrics 
    labels = metrics_dict['per_class']['class_labels']
    prec = metrics_dict['per_class']['precision_per_class']
    rec = metrics_dict['per_class']['recall_per_class']
    f1s = metrics_dict['per_class']['f1_per_class']

    for idx, label in enumerate(labels):
        row[f'{label}_precision'] = prec[idx]
        row[f'{label}_recall'] = rec[idx]
        row[f'{label}_f1'] = f1s[idx]

    return row

# =============================================================================
# function 4:
# =============================================================================
def evaluate_on_test(builder, y_pred):
    test_edges = builder.test_DTI['edges']

    drug_ids = builder.test_drugs_dic['ids']
    protein_ids = builder.test_targets_dic['ids']

    drug_id_to_index = {id_: idx for idx, id_ in enumerate(drug_ids)}
    prot_id_to_index = {id_: idx for idx, id_ in enumerate(protein_ids)}

    labels_list, d_list, p_list = [], [], []

    for _, row in test_edges.iterrows():
        d_id = row['DrugID']
        p_id = row['UniprotID']
        y = row['ActionLabel']
        if d_id in drug_id_to_index and p_id in prot_id_to_index:
            labels_list.append(y)
            d_list.append(d_id)
            p_list.append(p_id)

    results_df = pd.DataFrame({
        "DrugID": d_list,
        "UniprotID": p_list,
        "TrueLabel": labels_list,
        "PredLabel": y_pred
    })
    results_df.to_csv(os.path.join(cfg.run_dir, "test_predictions.csv"), index=False)

    print(" Evaluation completed and results saved.")


# =============================================================================
# function 5:
# =============================================================================  
def roc_auc(clf, y, X_train):
    score = metrics.roc_auc_score(y,
                                  clf.predict_proba(X_train),
                                  multi_class="ovr",
                                  average=cfg.metrics.average) if hasattr(clf, 'predict_proba') else np.nan
    return score


# =============================================================================
# function 6:      plot_roc_curve
# ============================================================================= 
def plot_roc_curve(y_test, y_pred, plot_suffix, plot_label=''):
    """Plot and save the ROC curve for each classifier."""
    # Handle binary or multiclass classification
    n_classes = len(np.unique(y_test))
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    y_score_bin = y_pred  #label_binarize(y_pred, classes=np.unique(y_pred))
    # Compute ROC curve and ROC AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    if n_classes > 2:
        # Multi-class case: Compute ROC curve and ROC AUC for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], y_score_bin[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    else:
        # Binary case: Compute ROC curve and ROC AUC
        fpr[0], tpr[0], _ = metrics.roc_curve(y_test, y_pred[:, 1])
        roc_auc[0] = metrics.auc(fpr[0], tpr[0])

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    if n_classes > 2:
        colors = cycle(['blue', 'red', 'green'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
    else:
        plt.plot(fpr[0], tpr[0], color='blue', lw=2,
                 label=f'ROC curve (area = {roc_auc[0]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {plot_suffix}')
    plt.legend(loc="lower right")

    # Save the figure
    os.makedirs(os.path.join(cfg.run_dir, plot_label), exist_ok=True)
    plt.savefig(f'{cfg.run_dir}/{plot_label}/roc_curve_{plot_suffix}.png')

    plt.close()

# =============================================================================
#  function 7
# =============================================================================
def plot_pr_curves(y_true, y_pred_proba, class_labels, plot_suffix, plot_label=''):
    # One-hot encode true labels for multiclass PR
    y_true_bin = label_binarize(y_true, classes=class_labels)
    plt.figure(figsize=(8, 6))
    for i, clf_label in enumerate(class_labels):
        precision, recall, _ = metrics.precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
        ap = metrics.average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
        plt.plot(recall, precision, label=f"{clf_label} (AP={ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {plot_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.join(cfg.run_dir, plot_label), exist_ok=True)
    plt.savefig(f'{cfg.run_dir}/{plot_label}/pr_curve_{plot_suffix}_{plot_label}.png')
    plt.close()

# =============================================================================
# function 8:     Loss_CategoricalCrossEntropy
# =============================================================================
def loss_categorical_cross_entropy(y_true, y_pred, one_hot=True):
    if not one_hot:
        encoder = OneHotEncoder(sparse_output=False)
        y_true_one_hot = y_true.reshape(-1, 1)
        y_true_one_hot = encoder.fit_transform(y_true_one_hot)

    else:
        y_true_one_hot = y_true
    # Clip data to prevent division by 0
    # Clip both sides to not drag mean towards any value
    y_pred_clipped = np.clip(y_pred, np.e ** -7, 1)  #, 1-e(1)**-7)

    # Probabilities for target values -
    # only if categorical labels
    if len(y_true_one_hot.shape) == 1:
        correct_confidences = y_pred_clipped * y_true_one_hot  #[range(samples), y_true_one_hot]
    # Mask values - only for one-hot encoded labels
    elif len(y_true_one_hot.shape) == 2:
        correct_confidences = np.sum(y_pred_clipped * y_true_one_hot, axis=1)
        # Losses
    negative_log_likelihoods = -np.log(correct_confidences)
    avg_loss = np.mean(negative_log_likelihoods)

    return avg_loss

#==============================================================================
#function 9:    calculate confusion matrix
# =============================================================================
# Plot Confusion Matrix
def confusion_mat(y_true, y_pred, suffix, cmdir):
    disp = metrics.ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=cfg.dataset.class_labels,
        cmap='Blues',
        normalize='true'
    )
    plt.title(f"Confusion Matrix - {suffix}")
    plt.grid(False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Save the plot
    os.makedirs(os.path.join(cfg.run_dir, cmdir), exist_ok=True)
    plt.savefig(f'{cfg.run_dir}/{cmdir}/confusion_matrix_{suffix}.png',
                dpi=300, bbox_inches='tight')

    # Show the plot (optional)
    plt.close()

# =============================================================================
# function 10:
# =============================================================================
def save_results(results, run_dir, clf_name, label=''):
    """
    Save the results to CSV files:
        - Full results per repetition and fold
        - Fold-wise mean/variance statistics
        - Repetition-wise mean/variance statistics
    """

    path = os.path.join(run_dir, label + '_' + clf_name)

    # Create DataFrame directly from the list of flattened dicts
    df = pd.DataFrame(results)

    # Save full results
    df.to_csv(path + '_results.csv', index=False)

    # Remove fold-specific columns safely if they exist
    drop_cols = ['Fold', 'drug_n_components', 'protein_n_components', 'Feature Combination Method']
    drop_cols_present = [col for col in drop_cols if col in df.columns]
    df_no_fold = df.drop(columns=drop_cols_present)

    # Compute fold-wise mean and variance
    group_cols = ['Classifier', 'repetition']
    group_cols_present = [col for col in group_cols if col in df_no_fold.columns]
    if group_cols_present:
        df_stats: DataFrame = df_no_fold.groupby(group_cols_present).agg(['mean', 'var']).reset_index()
        df_stats.to_csv(path + '_fold_stats.csv', index=False)

    # Compute repetition-wise mean and variance
    if 'repetition' in df_no_fold.columns:
        df_no_rep = df_no_fold.drop(columns=['repetition'])
    else:
        df_no_rep = df_no_fold.copy()

    group_cols_rep = ['Classifier'] if 'Classifier' in df_no_rep.columns else []
    if group_cols_rep:
        df_stats_mean = df_no_rep.groupby(group_cols_rep).agg(['mean', 'var']).reset_index()
        df_stats_mean.to_csv(path + '_fold_stats_mean.csv', index=False)

    return df
