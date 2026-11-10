"""Evaluation: per-class metrics, calibration, patient-cluster bootstrap CIs, McNemar, subgroups, temperature scaling."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..utils import NUM_CLASSES


def confusion(y: Sequence[int], p: Sequence[int], k: int = NUM_CLASSES) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (np.asarray(y), np.asarray(p)), 1)
    return cm


def per_class_from_cm(cm: np.ndarray) -> List[Dict[str, float]]:
    out = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        sup = cm[c].sum()
        pred = cm[:, c].sum()
        prec = tp / pred if pred else 0.0
        rec = tp / sup if sup else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.append({"precision": float(prec), "recall": float(rec), "f1": float(f1), "support": int(sup)})
    return out


def classification_report(y: Sequence[int], p: Sequence[int], k: int = NUM_CLASSES) -> Dict:
    cm = confusion(y, p, k)
    pc = per_class_from_cm(cm)
    n = cm.sum()
    return {
        "confusion": cm.tolist(),
        "accuracy": float(np.trace(cm) / n) if n else 0.0,
        "macro_f1": float(np.mean([c["f1"] for c in pc])),
        "balanced_accuracy": float(np.mean([c["recall"] for c in pc])),
        "per_class": pc,
    }


def reliability(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> Dict:
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    rows, ece, mce = [], 0.0, 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            rows.append({"lo": edges[b], "hi": edges[b + 1], "conf": 0.0, "acc": 0.0, "count": 0})
            continue
        c, a = conf[m].mean(), correct[m].mean()
        ece += m.mean() * abs(a - c)
        mce = max(mce, abs(a - c))
        rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "conf": float(c), "acc": float(a), "count": int(m.sum())})
    return {"bins": rows, "ece": float(ece), "mce": float(mce)}


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean())


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    oh = np.eye(probs.shape[1])[labels]
    return float(((probs - oh) ** 2).sum(1).mean())


def fit_temperature(val_probs: np.ndarray, val_labels: np.ndarray) -> float:
    """Temperature T minimising validation NLL of softmax(log p / T). Fit on validation, apply to test."""
    from scipy.optimize import minimize_scalar

    logp = np.log(np.clip(val_probs, 1e-12, 1))

    def f(t):
        z = logp / t
        z = z - z.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        return nll(p, val_labels)

    return float(minimize_scalar(f, bounds=(0.05, 10.0), method="bounded").x)


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(probs, 1e-12, 1)) / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def bootstrap_ci(labels: np.ndarray, preds: np.ndarray, groups: Optional[np.ndarray] = None, n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> Dict[str, List[float]]:
    """Percentile bootstrap for accuracy and macro-F1. If `groups` (patient ids) is given, resample *patients*
    (cluster bootstrap), which respects within-patient correlation and gives wider, more honest intervals."""
    rng = np.random.default_rng(seed)
    labels, preds = np.asarray(labels), np.asarray(preds)
    if groups is not None:
        ug, inv = np.unique(groups, return_inverse=True)
        members = [np.nonzero(inv == g)[0] for g in range(len(ug))]
    acc, f1 = [], []
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, len(labels), len(labels))
        else:
            pick = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[g] for g in pick])
        r = classification_report(labels[idx], preds[idx])
        acc.append(r["accuracy"])
        f1.append(r["macro_f1"])
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"accuracy": np.percentile(acc, q).tolist(), "macro_f1": np.percentile(f1, q).tolist(), "n_boot": n_boot, "cluster": groups is not None}


def mcnemar(labels: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    """Exact two-sided McNemar test on paired correctness (binomial on discordant pairs)."""
    from scipy.stats import binomtest

    ca, cb = pred_a == labels, pred_b == labels
    b = int((ca & ~cb).sum())
    c = int((~ca & cb).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"b_only_a_correct": b, "c_only_b_correct": c, "statistic": float(min(b, c)), "p_value": float(p)}


def by_group(labels: np.ndarray, preds: np.ndarray, groups: Iterable) -> Dict[str, Dict[str, float]]:
    groups = np.asarray(list(groups))
    out = {}
    for g in np.unique(groups):
        m = groups == g
        r = classification_report(labels[m], preds[m])
        out[str(g)] = {"n": int(m.sum()), "accuracy": r["accuracy"], "macro_f1": r["macro_f1"]}
    return out


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm–Bonferroni adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
