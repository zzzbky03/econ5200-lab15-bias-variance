"""Model selection whose reported score was never used to make the choice (ECON 5200 Lab 15).

The design rule: a function that chooses a model returns a nested score with its choice, never
the winning CV score.
"""
import numpy as np
from sklearn.model_selection import KFold


def cv_score(x, y, deg, k=5, seed=42):
    """Plain k-fold MSE of a degree-deg polynomial. Valid only for a degree fixed in advance."""
    errors = []
    for train, test in KFold(n_splits=k, shuffle=True, random_state=seed).split(x):
        coefs = np.polyfit(x[train], y[train], deg)
        errors.append(np.mean((np.polyval(coefs, x[test]) - y[test]) ** 2))
    return float(np.mean(errors))


def select(x, y, degrees, k=5, seed=42):
    """Choose a degree by k-fold CV, and score that choice by nested CV."""
    def choose(x_in, y_in):
        scores = [cv_score(x_in, y_in, d, k, seed) for d in degrees]
        return degrees[int(np.argmin(scores))]

    nested = []
    for train, test in KFold(n_splits=k, shuffle=True, random_state=seed).split(x):
        best = choose(x[train], y[train])
        coefs = np.polyfit(x[train], y[train], best)
        nested.append(np.mean((np.polyval(coefs, x[test]) - y[test]) ** 2))
    return {"choice": choose(x, y), "nested_score": float(np.mean(nested))}
