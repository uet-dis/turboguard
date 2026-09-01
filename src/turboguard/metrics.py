"""Comprehensive evaluation metrics for TurboGuard.

Provides individual metric functions and a ``compute_all_metrics()`` aggregate
that returns every relevant metric in a single dict. All percentage-scale
metrics use 0-100 range. Class 1 (attack) is the positive class throughout.

Categories:
    - Basic: accuracy, balanced accuracy
    - Binary classification: precision, recall, F1, specificity, FPR, FNR
    - Multi-average: micro, macro, weighted precision/recall/F1
    - Ranking: ROC-AUC, PR-AUC (average precision)
    - Correlation: MCC, Cohen's Kappa
    - Adversarial-specific: ADR, EDR, Filter-EDR, Clf-EDR
    - Advanced: G-Mean, Youden's J, log loss
    - Confusion matrix: TP, TN, FP, FN counts and rates
    - SHAP: feature importance via SHAP values (optional)
"""

from typing import Any, Dict, Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════
# Confusion matrix primitives
# ══════════════════════════════════════════════════════════════════════


def confusion_matrix_counts(preds: np.ndarray, y_true: np.ndarray) -> Dict[str, int]:
    """Computes raw confusion matrix counts.

    Args:
        preds: Binary predictions (0 or 1).
        y_true: Ground truth labels (0 = benign, 1 = attack).

    Returns:
        Dict with keys TP, TN, FP, FN as integer counts.
    """
    tp = int(((preds == 1) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


# ══════════════════════════════════════════════════════════════════════
# Basic classification metrics
# ══════════════════════════════════════════════════════════════════════


def compute_accuracy(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes overall classification accuracy (%).

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        Accuracy as percentage (0-100).
    """
    return float((preds == y_true).sum()) / max(len(y_true), 1) * 100


def compute_balanced_accuracy(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes balanced accuracy (%).

    Average of per-class recall, robust to class imbalance.
    ``BA = (TPR + TNR) / 2``

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        Balanced accuracy as percentage (0-100).
    """
    cm = confusion_matrix_counts(preds, y_true)
    tpr = cm["TP"] / max(cm["TP"] + cm["FN"], 1)
    tnr = cm["TN"] / max(cm["TN"] + cm["FP"], 1)
    return (tpr + tnr) / 2 * 100


# ══════════════════════════════════════════════════════════════════════
# Binary precision / recall / F-scores
# ══════════════════════════════════════════════════════════════════════


def compute_precision(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes precision for the positive class (attack) (%).

    ``Precision = TP / (TP + FP)``

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        Precision as percentage (0-100).
    """
    cm = confusion_matrix_counts(preds, y_true)
    denom = cm["TP"] + cm["FP"]
    return cm["TP"] / denom * 100 if denom > 0 else 0.0


def compute_recall(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes recall / sensitivity / TPR for attacks (%).

    ``Recall = TP / (TP + FN)``

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        Recall as percentage (0-100).
    """
    cm = confusion_matrix_counts(preds, y_true)
    denom = cm["TP"] + cm["FN"]
    return cm["TP"] / denom * 100 if denom > 0 else 0.0


def compute_f1(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes F1-score for the positive class (%).

    Harmonic mean of precision and recall.

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        F1 as percentage (0-100).
    """
    p = compute_precision(preds, y_true)
    r = compute_recall(preds, y_true)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def compute_f2(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes F2-score (%).

    Weights recall 2x higher than precision — useful when missing
    an attack (FN) is costlier than a false alarm (FP).

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        F2 as percentage (0-100).
    """
    p = compute_precision(preds, y_true)
    r = compute_recall(preds, y_true)
    beta2 = 4.0  # beta^2 = 2^2
    return (1 + beta2) * p * r / (beta2 * p + r) if (beta2 * p + r) > 0 else 0.0


def compute_f05(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes F0.5-score (%).

    Weights precision 2x higher than recall — useful when false
    alarms (FP) are costlier than missed attacks (FN).

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        F0.5 as percentage (0-100).
    """
    p = compute_precision(preds, y_true)
    r = compute_recall(preds, y_true)
    beta2 = 0.25  # beta^2 = 0.5^2
    return (1 + beta2) * p * r / (beta2 * p + r) if (beta2 * p + r) > 0 else 0.0


def compute_precision_recall_f1(
    preds: np.ndarray, y_true: np.ndarray
) -> tuple[float, float, float]:
    """Computes precision, recall, and F1-score (all %).

    Convenience wrapper returning all three in one call.

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        Tuple of (precision, recall, F1), each as percentage.
    """
    return (
        compute_precision(preds, y_true),
        compute_recall(preds, y_true),
        compute_f1(preds, y_true),
    )


# ══════════════════════════════════════════════════════════════════════
# Specificity and error rates
# ══════════════════════════════════════════════════════════════════════


def compute_specificity(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes specificity / True Negative Rate (%).

    ``TNR = TN / (TN + FP)``

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        Specificity as percentage (0-100).
    """
    cm = confusion_matrix_counts(preds, y_true)
    denom = cm["TN"] + cm["FP"]
    return cm["TN"] / denom * 100 if denom > 0 else 0.0


def compute_fpr(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes False Positive Rate (%).

    ``FPR = FP / (FP + TN) = 1 - Specificity``

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        FPR as percentage (0-100).
    """
    return 100.0 - compute_specificity(preds, y_true)


def compute_fnr(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes False Negative Rate / Miss Rate (%).

    ``FNR = FN / (FN + TP) = 1 - Recall``

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        FNR as percentage (0-100).
    """
    return 100.0 - compute_recall(preds, y_true)


# ══════════════════════════════════════════════════════════════════════
# Multi-average metrics (micro, macro, weighted)
# ══════════════════════════════════════════════════════════════════════


def _per_class_prf(preds: np.ndarray, y_true: np.ndarray) -> dict:
    """Computes per-class (0 and 1) precision, recall, F1, support.

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        Dict with keys 0 and 1, each containing P, R, F1, support.
    """
    result = {}
    for cls in [0, 1]:
        tp = int(((preds == cls) & (y_true == cls)).sum())
        fp = int(((preds == cls) & (y_true != cls)).sum())
        fn = int(((preds != cls) & (y_true == cls)).sum())
        support = int((y_true == cls).sum())
        p = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        result[cls] = {"precision": p, "recall": r, "f1": f1, "support": support}
    return result


def compute_macro_precision(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes macro-averaged precision (%).

    Unweighted mean of per-class precisions.

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        Macro precision as percentage.
    """
    prf = _per_class_prf(preds, y_true)
    return (prf[0]["precision"] + prf[1]["precision"]) / 2


def compute_macro_recall(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes macro-averaged recall (%).

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        Macro recall as percentage.
    """
    prf = _per_class_prf(preds, y_true)
    return (prf[0]["recall"] + prf[1]["recall"]) / 2


def compute_macro_f1(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes macro-averaged F1-score (%).

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        Macro F1 as percentage.
    """
    prf = _per_class_prf(preds, y_true)
    return (prf[0]["f1"] + prf[1]["f1"]) / 2


def compute_weighted_precision(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes support-weighted precision (%).

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        Weighted precision as percentage.
    """
    prf = _per_class_prf(preds, y_true)
    total = prf[0]["support"] + prf[1]["support"]
    if total == 0:
        return 0.0
    return (
        prf[0]["precision"] * prf[0]["support"] + prf[1]["precision"] * prf[1]["support"]
    ) / total


def compute_weighted_recall(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes support-weighted recall (%).

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        Weighted recall as percentage.
    """
    prf = _per_class_prf(preds, y_true)
    total = prf[0]["support"] + prf[1]["support"]
    if total == 0:
        return 0.0
    return (prf[0]["recall"] * prf[0]["support"] + prf[1]["recall"] * prf[1]["support"]) / total


def compute_weighted_f1(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes support-weighted F1-score (%).

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        Weighted F1 as percentage.
    """
    prf = _per_class_prf(preds, y_true)
    total = prf[0]["support"] + prf[1]["support"]
    if total == 0:
        return 0.0
    return (prf[0]["f1"] * prf[0]["support"] + prf[1]["f1"] * prf[1]["support"]) / total


def compute_micro_f1(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes micro-averaged F1-score (%).

    Micro-averaging aggregates TP/FP/FN across all classes before
    computing the metric. For binary classification, micro-F1 equals
    accuracy.

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        Micro F1 as percentage (equals accuracy for binary).
    """
    return compute_accuracy(preds, y_true)


# ══════════════════════════════════════════════════════════════════════
# Correlation and agreement metrics
# ══════════════════════════════════════════════════════════════════════


def compute_mcc(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes Matthews Correlation Coefficient.

    MCC ranges from -1 (total disagreement) to +1 (perfect). Returns
    the raw coefficient (not percentage) since its scale is -1 to 1.

    MCC is considered one of the most informative single metrics for
    binary classification, especially under class imbalance.

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        MCC as a float in [-1, 1].
    """
    cm = confusion_matrix_counts(preds, y_true)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def compute_cohens_kappa(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes Cohen's Kappa coefficient.

    Measures agreement between predictions and ground truth, adjusted
    for chance agreement. Ranges from -1 to 1 (1 = perfect, 0 = chance).

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        Kappa as a float in [-1, 1].
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    cm = confusion_matrix_counts(preds, y_true)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]

    po = (tp + tn) / n  # Observed agreement
    # Expected agreement under independence.
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


# ══════════════════════════════════════════════════════════════════════
# Geometric and composite metrics
# ══════════════════════════════════════════════════════════════════════


def compute_gmean(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes Geometric Mean of sensitivity and specificity (%).

    ``G-Mean = sqrt(TPR × TNR)``

    Robust to class imbalance — both sensitivity and specificity must
    be high for a good score.

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        G-Mean as percentage (0-100).
    """
    tpr = compute_recall(preds, y_true) / 100
    tnr = compute_specificity(preds, y_true) / 100
    return np.sqrt(tpr * tnr) * 100


def compute_youdens_j(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes Youden's J statistic.

    ``J = TPR + TNR - 1 = Sensitivity + Specificity - 1``

    Ranges from -1 to 1. Commonly used to select optimal thresholds.

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        Youden's J as a float in [-1, 1].
    """
    tpr = compute_recall(preds, y_true) / 100
    tnr = compute_specificity(preds, y_true) / 100
    return tpr + tnr - 1


# ══════════════════════════════════════════════════════════════════════
# Ranking / probabilistic metrics (require scores, not just predictions)
# ══════════════════════════════════════════════════════════════════════


def compute_roc_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    """Computes ROC Area Under Curve (%).

    Requires continuous anomaly scores or predicted probabilities,
    not binary predictions.

    Args:
        y_true: Ground truth binary labels.
        y_scores: Continuous scores (higher = more likely attack).

    Returns:
        ROC-AUC as percentage (0-100). Returns 50.0 if sklearn unavailable.
    """
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_scores)) * 100
    except (ImportError, ValueError):
        return 50.0


def compute_pr_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    """Computes Precision-Recall AUC / Average Precision (%).

    More informative than ROC-AUC under severe class imbalance (which
    is typical for NIDS datasets).

    Args:
        y_true: Ground truth binary labels.
        y_scores: Continuous scores (higher = more likely attack).

    Returns:
        PR-AUC as percentage (0-100).
    """
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y_true, y_scores)) * 100
    except (ImportError, ValueError):
        return 0.0


def compute_log_loss(
    y_true: np.ndarray,
    y_probs: np.ndarray,
) -> float:
    """Computes Binary Cross-Entropy / Log Loss.

    Lower is better. Requires predicted probabilities, not binary labels.

    Args:
        y_true: Ground truth binary labels.
        y_probs: Predicted probability of class 1.

    Returns:
        Log loss as a float (lower is better).
    """
    try:
        from sklearn.metrics import log_loss

        return float(log_loss(y_true, y_probs))
    except (ImportError, ValueError):
        return float("inf")


def compute_brier_score(
    y_true: np.ndarray,
    y_probs: np.ndarray,
) -> float:
    """Computes Brier Score (mean squared error of probabilities).

    ``BS = mean((p - y)^2)``

    Measures calibration quality. Lower is better. Perfect = 0.

    Args:
        y_true: Ground truth binary labels.
        y_probs: Predicted probability of class 1.

    Returns:
        Brier score as a float in [0, 1].
    """
    return float(np.mean((y_probs - y_true) ** 2))


# ══════════════════════════════════════════════════════════════════════
# Adversarial-specific metrics
# ══════════════════════════════════════════════════════════════════════


def compute_adr(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes Attack Detection Rate (%).

    ``ADR = TP / (TP + FN)`` — equivalent to recall, but named for
    the adversarial detection context.

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels (0 = benign, 1 = attack).

    Returns:
        ADR as percentage (0-100).
    """
    attack = y_true == 1
    if attack.sum() == 0:
        return 0.0
    return float((preds[attack] == 1).sum()) / int(attack.sum()) * 100


def compute_edr(preds: np.ndarray, is_evasion: np.ndarray) -> float:
    """Computes Evasion Detection Rate (%).

    Measures what fraction of adversarially-perturbed samples are
    still correctly detected as attacks.

    Args:
        preds: Binary predictions.
        is_evasion: Boolean mask for adversarially-modified samples.

    Returns:
        EDR as percentage (0-100).
    """
    n = is_evasion.sum()
    if n == 0:
        return 0.0
    return float((preds[is_evasion] == 1).sum()) / int(n) * 100


def compute_filter_edr(
    iso_blocked: np.ndarray,
    is_evasion: np.ndarray,
) -> float:
    """Computes Filter EDR — fraction caught by the IF hard-drop stage (%).

    Measures the IsolationForest's standalone contribution to evasion
    detection, independent of the DNN grey-zone classifier.

    Args:
        iso_blocked: Boolean mask of samples blocked by IsolationForest.
        is_evasion: Boolean mask for adversarially-modified samples.

    Returns:
        Filter-EDR as percentage (0-100).
    """
    n = is_evasion.sum()
    if n == 0:
        return 0.0
    return float((iso_blocked & is_evasion).sum()) / int(n) * 100


def compute_clf_edr(
    preds: np.ndarray,
    iso_blocked: np.ndarray,
    is_evasion: np.ndarray,
) -> float:
    """Computes Classifier EDR — fraction caught by the DNN grey-zone (%).

    Measures what the DNN classifier catches among evasion samples
    that passed through the IsolationForest filter.

    Args:
        preds: Final binary predictions (combined IF + DNN).
        iso_blocked: Boolean mask of samples blocked by IsolationForest.
        is_evasion: Boolean mask for adversarially-modified samples.

    Returns:
        Clf-EDR as percentage (0-100).
    """
    # Evasion samples that were NOT blocked by IF but still detected.
    grey_evasion = is_evasion & ~iso_blocked
    n = grey_evasion.sum()
    if n == 0:
        return 0.0
    return float((preds[grey_evasion] == 1).sum()) / int(n) * 100


def compute_attack_success_rate(preds: np.ndarray, y_true: np.ndarray) -> float:
    """Computes Attack Success Rate (ASR) (%).

    ``ASR = FN / (TP + FN)`` — fraction of attacks that evaded
    detection. ``ASR = 1 - ADR``.

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.

    Returns:
        ASR as percentage (0-100). Lower is better for the defender.
    """
    return 100.0 - compute_adr(preds, y_true)


# ══════════════════════════════════════════════════════════════════════
# Per-class classification report
# ══════════════════════════════════════════════════════════════════════


def classification_report(
    preds: np.ndarray,
    y_true: np.ndarray,
    class_names: tuple[str, str] = ("Benign", "Attack"),
) -> Dict[str, Dict[str, float]]:
    """Generates a per-class classification report.

    Similar to sklearn's ``classification_report`` but returns a dict
    with percentage-scale metrics.

    Args:
        preds: Binary predictions.
        y_true: Ground truth labels.
        class_names: Names for class 0 and class 1.

    Returns:
        Nested dict: ``{class_name: {precision, recall, f1, support}}``.
    """
    prf = _per_class_prf(preds, y_true)
    return {
        class_names[0]: prf[0],
        class_names[1]: prf[1],
    }


# ══════════════════════════════════════════════════════════════════════
# SHAP feature importance
# ══════════════════════════════════════════════════════════════════════


def compute_shap_importance(
    model: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: Optional[list[str]] = None,
    max_background: int = 200,
    max_explain: int = 500,
) -> Dict[str, Any]:
    """Computes SHAP feature importance for a model.

    Uses DeepExplainer for PyTorch models and KernelExplainer as
    fallback. Returns mean absolute SHAP values per feature (global
    importance) and optionally the raw SHAP matrix.

    Requires ``shap`` to be installed (``pip install shap``).

    Args:
        model: Trained model (PyTorch nn.Module, sklearn estimator, or
            any callable that returns predictions).
        X_background: Background/reference samples for SHAP.
        X_explain: Samples to explain.
        feature_names: Optional feature name list.
        max_background: Max background samples (subsampled for speed).
        max_explain: Max samples to explain.

    Returns:
        Dict with keys:
            - mean_abs_shap: Per-feature mean |SHAP| values (np.ndarray).
            - feature_names: Feature name list.
            - feature_importance: Sorted list of (name, importance) tuples.
            - shap_values: Raw SHAP matrix if available.
    """
    try:
        import shap
        import torch
        import torch.nn as nn
    except ImportError:
        return {
            "error": "shap and/or torch not installed",
            "mean_abs_shap": np.array([]),
            "feature_names": feature_names or [],
            "feature_importance": [],
        }

    n_bg = min(max_background, len(X_background))
    n_ex = min(max_explain, len(X_explain))
    bg = X_background[np.random.choice(len(X_background), n_bg, replace=False)]
    ex = X_explain[np.random.choice(len(X_explain), n_ex, replace=False)]

    shap_values = None

    if isinstance(model, nn.Module):
        # PyTorch model — use DeepExplainer.
        model.eval()
        bg_t = torch.tensor(bg, dtype=torch.float32)
        ex_t = torch.tensor(ex, dtype=torch.float32)
        device = next(model.parameters()).device
        bg_t = bg_t.to(device)
        ex_t = ex_t.to(device)

        try:
            explainer = shap.DeepExplainer(model, bg_t)
            shap_values = explainer.shap_values(ex_t)
        except Exception:
            # Fallback to GradientExplainer if DeepExplainer fails
            # (e.g. due to BatchNorm or unsupported ops).
            try:
                explainer = shap.GradientExplainer(model, bg_t)
                shap_values = explainer.shap_values(ex_t)
            except Exception:
                pass

        # DeepExplainer returns list for multi-output; take class 1.
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]
        if hasattr(shap_values, "cpu"):
            shap_values = shap_values.cpu().numpy()
    else:
        # Sklearn or other — use KernelExplainer.
        try:
            if hasattr(model, "predict_proba"):
                explainer = shap.KernelExplainer(model.predict_proba, bg)
            else:
                explainer = shap.KernelExplainer(model.predict, bg)
            shap_values = explainer.shap_values(ex)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]
        except Exception:
            pass

    if shap_values is None:
        return {
            "error": "SHAP computation failed",
            "mean_abs_shap": np.array([]),
            "feature_names": feature_names or [],
            "feature_importance": [],
        }

    shap_values = np.array(shap_values)
    mean_abs = np.mean(np.abs(shap_values), axis=0)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(mean_abs))]

    # Sort by importance descending.
    sorted_idx = np.argsort(mean_abs)[::-1]
    importance = [(feature_names[i], float(mean_abs[i])) for i in sorted_idx]

    return {
        "mean_abs_shap": mean_abs,
        "feature_names": feature_names,
        "feature_importance": importance,
        "shap_values": shap_values,
    }


# ══════════════════════════════════════════════════════════════════════
# Aggregate: compute ALL metrics in one call
# ══════════════════════════════════════════════════════════════════════


def compute_all_metrics(
    preds: np.ndarray,
    y_true: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
    y_probs: Optional[np.ndarray] = None,
    is_evasion: Optional[np.ndarray] = None,
    iso_blocked: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Computes every available metric in a single call.

    Returns a flat dict of metric_name → value. Percentage-scale
    metrics are in 0-100 range; correlation metrics (MCC, Kappa, J)
    are in their native scale.

    Args:
        preds: Binary predictions (0 or 1).
        y_true: Ground truth binary labels.
        y_scores: Optional continuous anomaly scores for ROC/PR AUC.
        y_probs: Optional predicted probabilities for log loss / Brier.
        is_evasion: Optional boolean mask for EDR computation.
        iso_blocked: Optional boolean mask for Filter/Clf EDR.

    Returns:
        Dict of all computed metrics.
    """
    results: Dict[str, float] = {}
    cm = confusion_matrix_counts(preds, y_true)

    # Confusion matrix.
    results["TP"] = cm["TP"]
    results["TN"] = cm["TN"]
    results["FP"] = cm["FP"]
    results["FN"] = cm["FN"]

    # Basic.
    results["Accuracy"] = compute_accuracy(preds, y_true)
    results["Balanced_Accuracy"] = compute_balanced_accuracy(preds, y_true)

    # Binary classification (attack = positive).
    results["Precision"] = compute_precision(preds, y_true)
    results["Recall"] = compute_recall(preds, y_true)
    results["F1"] = compute_f1(preds, y_true)
    results["F2"] = compute_f2(preds, y_true)
    results["F0.5"] = compute_f05(preds, y_true)

    # Rates.
    results["Specificity"] = compute_specificity(preds, y_true)
    results["FPR"] = compute_fpr(preds, y_true)
    results["FNR"] = compute_fnr(preds, y_true)

    # Multi-average.
    results["Macro_Precision"] = compute_macro_precision(preds, y_true)
    results["Macro_Recall"] = compute_macro_recall(preds, y_true)
    results["Macro_F1"] = compute_macro_f1(preds, y_true)
    results["Weighted_Precision"] = compute_weighted_precision(preds, y_true)
    results["Weighted_Recall"] = compute_weighted_recall(preds, y_true)
    results["Weighted_F1"] = compute_weighted_f1(preds, y_true)
    results["Micro_F1"] = compute_micro_f1(preds, y_true)

    # Correlation.
    results["MCC"] = compute_mcc(preds, y_true)
    results["Cohens_Kappa"] = compute_cohens_kappa(preds, y_true)

    # Geometric / composite.
    results["G_Mean"] = compute_gmean(preds, y_true)
    results["Youdens_J"] = compute_youdens_j(preds, y_true)

    # Adversarial-specific.
    results["ADR"] = compute_adr(preds, y_true)
    results["ASR"] = compute_attack_success_rate(preds, y_true)

    if is_evasion is not None:
        results["EDR"] = compute_edr(preds, is_evasion)
        if iso_blocked is not None:
            results["Filter_EDR"] = compute_filter_edr(iso_blocked, is_evasion)
            results["Clf_EDR"] = compute_clf_edr(preds, iso_blocked, is_evasion)

    # Ranking metrics (require continuous scores).
    if y_scores is not None:
        results["ROC_AUC"] = compute_roc_auc(y_true, y_scores)
        results["PR_AUC"] = compute_pr_auc(y_true, y_scores)

    # Probabilistic metrics (require predicted probabilities).
    if y_probs is not None:
        results["Log_Loss"] = compute_log_loss(y_true, y_probs)
        results["Brier_Score"] = compute_brier_score(y_true, y_probs)

    return results
