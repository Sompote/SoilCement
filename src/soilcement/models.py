"""In-context foundation models: the corpus and the trial mixes are the context, the query mixes are predicted
in one forward pass. Nothing is trained at the project."""
import os
import numpy as np


def _tabicl(checkpoint=None, device="cpu", n_estimators=2):
    try:
        from tabicl import TabICLRegressor
    except ImportError as e:
        raise SystemExit("TabICL is not installed: pip install 'soilcement[tabicl]'") from e
    kw = {"model_path": checkpoint} if checkpoint else {}
    return TabICLRegressor(device=device, n_estimators=n_estimators, random_state=0, **kw)


def _tabpfn3(device="cpu", n_estimators=2):
    try:
        from tabpfn import TabPFNRegressor
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit("TabPFN is not installed: pip install 'soilcement[tabpfn]'") from e
    ck = hf_hub_download("Prior-Labs/tabpfn_3", "tabpfn-v3-regressor-v3_default.ckpt")
    return TabPFNRegressor(model_path=ck, device=device, n_estimators=n_estimators, random_state=0)


def predict_in_context(name, X_ctx, y_ctx, X_query, device="cpu", checkpoint=None, alphas=(0.1, 0.9)):
    """name: tabicl (released TabICL v2), tabicl-soil (TabICL v2 pretrained further on the soil-cement prior)
    or tabpfn3. Returns mean, lower and upper quantiles of ln qu."""
    if name == "tabicl-soil":
        checkpoint = checkpoint or os.environ.get("SOILCEMENT_TABICL_SOIL")
        if not checkpoint or not os.path.exists(checkpoint):
            raise SystemExit("tabicl-soil needs the checkpoint: --checkpoint models/tabicl_soil.ckpt "
                             "(or set SOILCEMENT_TABICL_SOIL); see README, 'Model files'")
    if name in ("tabicl", "tabicl-soil"):
        m = _tabicl(checkpoint if name == "tabicl-soil" else None, device)
        m.fit(X_ctx, y_ctx)
        out = m.predict(X_query, output_type=["mean", "quantiles"], alphas=list(alphas))
        q = np.asarray(out["quantiles"])
        q = q if q.shape[0] == len(X_query) else q.T
        return np.asarray(out["mean"]), q[:, 0], q[:, 1]
    if name == "tabpfn3":
        m = _tabpfn3(device); m.fit(X_ctx, y_ctx)
        out = m.predict(X_query, output_type="full", quantiles=list(alphas))
        return np.asarray(out["mean"]), np.asarray(out["quantiles"][0]), np.asarray(out["quantiles"][1])
    raise SystemExit(f"unknown model {name}")
