"""Hierarchical strength law (Section 4.1 of the paper), with an exact grid posterior over the two
deposit latents, the held-water share phi and the strength offset delta.

ln qu = alpha - beta [ln((1 - phi) Cw/100) - a ln(Ac/100) - b ln(t/28)] + delta + eps,
phi ~ N(c max(0, wL - 0.6), s_phi) on [0, 0.9],  delta ~ N(0, s_delta),  eps ~ N(0, s_y)."""
import json
from importlib import resources
import numpy as np
from scipy.special import logsumexp

PHI = np.linspace(0, 0.9, 46)
DEL = np.linspace(-2.0, 2.0, 81)


class HierarchicalLaw:
    def __init__(self, constants=None):
        if constants is None:
            with resources.files("soilcement").joinpath("data/law_constants.json").open() as fh:
                constants = json.load(fh)
        self.k = constants

    def _mean(self, X, phi, delta=0.0):
        k = self.k; lnCw, lnAc, lnt = X[:, 0] + np.log(1.0), X[:, 1], X[:, 2]
        return k["alpha"] - k["beta"] * (np.log(1 - phi) + lnCw - k["a"] * lnAc - k["b"] * lnt) + delta

    def predict(self, X_trial, y_trial, X_query, wL, alphas=(0.1, 0.9), seed=0):
        """X_* are feature matrices from soilcement.features; y_trial = ln qu of the trial mixes (may be empty).
        Returns mean, lower and upper quantiles of ln qu for the query mixes."""
        k = self.k
        prior_phi = float(np.clip(k["c"] * max(0.0, wL / 100 - k.get("wL_threshold", 0.6)), 0, 0.9))
        lp = -0.5 * ((PHI[:, None] - prior_phi) / k["s_phi"]) ** 2 - 0.5 * (DEL[None, :] / k["s_delta"]) ** 2
        if len(y_trial):
            for i, ph in enumerate(PHI):
                r = y_trial[None, :] - self._mean(X_trial, ph)[None, :] - DEL[:, None]
                lp[i, :] += -0.5 * np.sum(r ** 2, axis=1) / k["s_y"] ** 2
        w = np.exp(lp - logsumexp(lp)).ravel()
        preds = np.array([self._mean(X_query, ph, d) for ph in PHI for d in DEL])
        mean = w @ preds
        rng = np.random.default_rng(seed); draws = rng.choice(len(w), size=2000, p=w)
        samp = preds[draws] + rng.normal(0, k["s_y"], (2000, preds.shape[1]))
        lo, hi = np.percentile(samp, [100 * alphas[0], 100 * alphas[1]], axis=0)
        return mean, lo, hi
