"""Classifier two-sample test (C2ST): can a classifier tell NPE samples from reference samples?

Accuracy 0.5 = indistinguishable; 1.0 = completely different. Same recipe as the sbi package
(standardise, MLP, 5-fold cross-validated accuracy).
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def c2st_accuracy(x_p: np.ndarray, x_q: np.ndarray, seed: int = 0, n_folds: int = 5) -> float:
    """Cross-validated accuracy for separating samples x_p (n, D) from x_q (m, D)."""
    x = np.concatenate([x_p, x_q])
    y = np.concatenate([np.zeros(len(x_p)), np.ones(len(x_q))])
    dim = x.shape[1]
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(10 * dim, 10 * dim),
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=50,
            random_state=seed,
        ),
    )
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(clf, x, y, cv=cv, scoring="accuracy")))
