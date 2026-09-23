import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def evaluate_model(y_true, y_probability, threshold=0.5):
    """
    Evaluates the model predictions based on a given threshold.
    Returns a dictionary of metrics.
    """
    y_pred = np.where(y_probability >= threshold, "Yes", "No")
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="Yes", zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label="Yes", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="Yes", zero_division=0)
    roc_auc = roc_auc_score((y_true == "Yes").astype(int), y_probability)
    cm = confusion_matrix(y_true, y_pred, labels=["No", "Yes"])
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist()
    }

