"""
Shared utilities (AUC/ROC, MC-dropout variance, partial correlation,
temperature scaling) used by gradient_effect_baselines.py,
gradient_effect_adversarial.py, and gradient_effect_adversarial_extended.py.
No paper figure/table of its own -- see Sections 5.5/5.6/5.7.
"""

import numpy as np
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────
# Cheap confidence-derived baselines
# ─────────────────────────────────────────────────────────────────────────

def softmax_entropy(probs_full, eps=1e-12):
    """Shannon entropy of the full softmax distribution, per sample.
    probs_full: (N, C) array. Returns (N,) array, nats.
    Only genuinely distinct from CI/margin when C > 2 (see module docstring)."""
    probs_full = np.clip(probs_full, eps, 1.0)
    return -(probs_full * np.log(probs_full)).sum(axis=1)


def top2_margin(probs_full):
    """Top1 - Top2 probability gap, per sample. (N, C) -> (N,).
    Rank-equivalent to CI for binary classifiers; used mainly as the
    temperature-scaling rank-equivalence sanity check."""
    sorted_probs = np.sort(probs_full, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def temperature_scale_logits(logits, T):
    """Apply scalar temperature T to logits, return softmax probs."""
    scaled = logits / T
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    e = np.exp(scaled)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits_cal, labels_cal, T_grid=None):
    """Fit a scalar temperature by grid search minimizing NLL on a
    calibration split (avoids needing autograd for a 1-D optimization)."""
    if T_grid is None:
        T_grid = np.concatenate([np.linspace(0.3, 3.0, 55), np.linspace(3.0, 10.0, 15)])
    n = len(labels_cal)
    best_T, best_nll = 1.0, np.inf
    for T in T_grid:
        probs = temperature_scale_logits(logits_cal, T)
        nll = -np.log(np.clip(probs[np.arange(n), labels_cal], 1e-12, 1.0)).mean()
        if nll < best_nll:
            best_nll, best_T = nll, T
    return best_T


# ─────────────────────────────────────────────────────────────────────────
# MC-dropout predictive variance (the genuinely distinct baseline)
# ─────────────────────────────────────────────────────────────────────────

def mc_dropout_variance(stochastic_predict_fn, x, n_passes=20):
    """
    stochastic_predict_fn(x) -> (B, C) array of softmax probs for ONE
    stochastic forward pass (caller is responsible for enabling dropout in
    whatever way is idiomatic for their architecture -- model.train() for
    PyTorch, dropout=True for the hand-rolled EEG MLP).

    Returns: (B,) variance of the top-class (argmax of the MEAN distribution)
    probability across n_passes stochastic passes. This is the standard
    "predictive variance" MC-dropout uncertainty score (Gal & Ghahramani 2016).
    """
    passes = np.stack([stochastic_predict_fn(x) for _ in range(n_passes)], axis=0)  # (T, B, C)
    mean_probs = passes.mean(axis=0)             # (B, C)
    top_cls = mean_probs.argmax(axis=1)           # (B,)
    top_probs_per_pass = passes[:, np.arange(len(top_cls)), top_cls]  # (T, B)
    return top_probs_per_pass.var(axis=0)          # (B,)


# ─────────────────────────────────────────────────────────────────────────
# ROC / AUC (promoted from gradient_effect_adversarial.py, unchanged logic)
# ─────────────────────────────────────────────────────────────────────────

def compute_roc_auc(scores, labels):
    """labels: 0/1 array, 1=positive class. Higher score = more positive."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    thresholds = np.sort(np.unique(scores))[::-1]
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None, None, float('nan')
    fpr_list, tpr_list = [0.0], [0.0]
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fpr_list.append(fp / neg)
        tpr_list.append(tp / pos)
    fpr_list.append(1.0)
    tpr_list.append(1.0)
    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    trapfn = getattr(np, 'trapezoid', None) or np.trapz
    return fpr, tpr, float(trapfn(tpr, fpr))


def best_direction_auc(scores, labels):
    """Some scores' "positive" direction is ambiguous a priori (e.g. raw CI
    for OOD detection could point either way depending on the OOD source).
    Returns the AUC using whichever sign gives AUC >= 0.5."""
    _, _, auc = compute_roc_auc(scores, labels)
    if np.isnan(auc):
        return auc
    if auc < 0.5:
        _, _, auc = compute_roc_auc(-scores, labels)
    return auc


# ─────────────────────────────────────────────────────────────────────────
# Partial correlation (item 8: confound control)
# ─────────────────────────────────────────────────────────────────────────

def partial_correlation(x, y, z):
    """
    Partial Pearson correlation of x and y, controlling for z.
    r_xy.z = (r_xy - r_xz*r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
    Returns (r_partial, p_value) using the standard t-test with n-3 df.
    """
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    n = len(x)
    r_xy, _ = stats.pearsonr(x, y)
    r_xz, _ = stats.pearsonr(x, z)
    r_yz, _ = stats.pearsonr(y, z)
    denom = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    r_partial = (r_xy - r_xz * r_yz) / denom if denom > 1e-12 else np.nan
    df = n - 3
    if df <= 0 or not np.isfinite(r_partial) or abs(r_partial) >= 1.0:
        return r_partial, np.nan
    t_stat = r_partial * np.sqrt(df / (1 - r_partial ** 2))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    return r_partial, p_value
