"""The corpus and the feature map shared by every model."""
from importlib import resources
import numpy as np
import pandas as pd

BINDER_FACTOR = {"opc": 1.0, "pcc": 1.0, "slag": 1.72}   # equivalent-cement factor by binder class
POZZOLAN_FACTOR = 0.75                                   # a unit of pozzolan counts as 0.75 of cement


def load_corpus() -> pd.DataFrame:
    """The main set used in the paper: 1,372 mixes from 38 deposits, replicates averaged."""
    with resources.files("soilcement").joinpath("data/corpus_main.csv").open() as fh:
        return pd.read_csv(fh)


def equivalent_cement(binder_pct, binder="opc", pozzolan_pct_of_cement=0.0):
    """Equivalent cement content A_c (% of dry clay) from the binder content (% of dry clay), the binder class
    (opc: Portland cement; pcc: Portland composite cement; slag: blast-furnace slag cement) and any pozzolan
    (fly ash, rice-husk ash, ...) added as a percentage of the cement."""
    lam = np.array([BINDER_FACTOR[str(b).lower()] for b in np.broadcast_to(np.asarray(binder, dtype=object), np.shape(binder_pct))])
    return lam * np.asarray(binder_pct, float) * (1 + POZZOLAN_FACTOR * np.asarray(pozzolan_pct_of_cement, float) / 100)


def features(water_content_pct, equivalent_cement_pct, curing_days, LL_pct, PI_pct, PL_pct=None):
    """The six model features: ln(Cw/100), ln(Ac/100), ln(t/28), wL, Ip, wP (limits as fractions)."""
    Cw, Ac, t = (np.asarray(v, float) for v in (water_content_pct, equivalent_cement_pct, curing_days))
    n = len(np.atleast_1d(Cw))
    LL = np.broadcast_to(np.asarray(LL_pct, float), (n,)); PI = np.broadcast_to(np.asarray(PI_pct, float), (n,))
    PL = LL - PI if PL_pct is None else np.broadcast_to(np.asarray(PL_pct, float), (n,))
    if np.any(Cw <= 0) or np.any(Ac <= 0) or np.any(t <= 0):
        raise ValueError("water content, cement content and curing time must be positive")
    return np.column_stack([np.log(Cw / 100), np.log(Ac / 100), np.log(t / 28), LL / 100, PI / 100, PL / 100])


def corpus_xy(df=None):
    df = load_corpus() if df is None else df
    X = features(df.water_content_pct, df.equivalent_cement_pct, df.curing_days, df.LL_pct, df.PI_pct, df.PL_pct)
    return X, np.log(df.qu_kPa.to_numpy(float)), df.deposit.to_numpy()
