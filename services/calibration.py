"""Score calibration: raw cosine similarity -> estimated probability of a correct mated match.

The calibrator is a plain JSON file written by `python -m eval run eval/config/calibration.yaml`:
  {"type": "isotonic", "x": [...], "y": [...], "prior": 0.1, ...}   piecewise-linear (np.interp)
  {"type": "platt", "a": ..., "b": ..., "prior": 0.1, ...}           p = 1 / (1 + exp(-(a*s + b)))

The probability depends on the share of mated searches assumed when fitting (`prior`). To
re-target another prior pi', apply Bayes' rule on the odds:
  odds' = odds * (pi' / (1 - pi')) / (pi / (1 - pi)),   p' = odds' / (1 + odds')
"""
import json
import math

import numpy as np


def adjust_prior(p, fitted_prior, target_prior):
    if target_prior is None or abs(target_prior - fitted_prior) < 1e-12:
        return p
    p = np.clip(np.asarray(p, dtype="float64"), 1e-9, 1 - 1e-9)
    odds = p / (1 - p)
    factor = (target_prior / (1 - target_prior)) / (fitted_prior / (1 - fitted_prior))
    new = odds * factor
    return new / (1 + new)


class Calibrator:
    def __init__(self, spec):
        self.spec = spec
        self.kind = spec["type"]
        self.prior = float(spec.get("prior", 0.1))
        if self.kind == "isotonic":
            self.x = np.asarray(spec["x"], dtype="float64")
            self.y = np.asarray(spec["y"], dtype="float64")
        elif self.kind == "platt":
            self.a, self.b = float(spec["a"]), float(spec["b"])
        else:
            raise ValueError(f"unknown calibrator type {self.kind}")

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def predict_many(self, scores, prior=None):
        s = np.asarray(scores, dtype="float64")
        if self.kind == "isotonic":
            p = np.interp(s, self.x, self.y)
        else:
            p = 1.0 / (1.0 + np.exp(-(self.a * s + self.b)))
        p = adjust_prior(p, self.prior, prior)
        return np.clip(p, 0.0, 1.0)

    def predict(self, score, prior=None):
        """Estimated P(correct mated match) for one rank-1 cosine score."""
        if score is None or (isinstance(score, float) and math.isnan(score)):
            return None
        return float(self.predict_many([score], prior)[0])
