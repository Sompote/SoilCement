"""Command-line interface.

  soilcement predict --ll 103 --pi 60 --trials trials.csv --query query.csv [--model auto] [--out pred.csv]
  soilcement design  --ll 103 --pi 60 --trials trials.csv --target 800 --water 150 --curing 28
  soilcement corpus  [--export corpus.csv]
"""
import argparse, sys
import numpy as np
import pandas as pd
from . import __version__
from .corpus import load_corpus, equivalent_cement, features, corpus_xy
from .law import HierarchicalLaw
from .models import predict_in_context

# Benchmark error on the 26 deposits scored at every k (Table 8 of the paper), per model:
# mean / median absolute percentage error and share within a factor of two.
TYPICAL = {'law': {0: (78, 45, 68), 1: (60, 33, 78), 2: (49, 31, 79), 3: (56, 30, 79), 5: (50, 29, 79), 8: (48, 28, 81), 12: (52, 29, 80)}, 'tabpfn3': {0: (134, 70, 46), 1: (72, 38, 69), 2: (47, 29, 77), 3: (41, 24, 82), 5: (37, 22, 87), 8: (28, 17, 91), 12: (29, 15, 92)}, 'tabicl': {0: (116, 58, 58), 1: (74, 39, 72), 2: (50, 28, 80), 3: (45, 25, 85), 5: (39, 22, 88), 8: (30, 17, 92), 12: (30, 16, 92)}, 'tabicl-soil': {0: (108, 52, 61), 1: (76, 34, 75), 2: (46, 26, 82), 3: (42, 23, 87), 5: (36, 19, 89), 8: (29, 16, 92), 12: (29, 15, 92)}}
NEED = {"water_content_pct", "binder_pct", "curing_days"}


def _mixes(path, need_qu):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    miss = (NEED | ({"qu_kPa"} if need_qu else set())) - set(df.columns)
    if miss:
        raise SystemExit(f"{path}: missing column(s) {sorted(miss)}; required: water_content_pct, binder_pct, curing_days"
                         + (", qu_kPa" if need_qu else "") + "; optional: binder (opc|pcc|slag), pozzolan_pct_of_cement")
    df["binder"] = df["binder"].fillna("opc") if "binder" in df else "opc"
    df["pozzolan_pct_of_cement"] = df["pozzolan_pct_of_cement"].fillna(0) if "pozzolan_pct_of_cement" in df else 0.0
    df["equivalent_cement_pct"] = equivalent_cement(df.binder_pct, df.binder, df.pozzolan_pct_of_cement)
    return df


def _X(df, a):
    return features(df.water_content_pct, df.equivalent_cement_pct, df.curing_days, a.ll, a.pi, a.pl)


def _predict(a, trials, query):
    k = 0 if trials is None else len(trials)
    model = a.model if a.model != "auto" else ("law" if k <= 1 else "tabicl")
    Xq = _X(query, a)
    if model == "law":
        Xt = _X(trials, a) if k else np.zeros((0, 6)); yt = np.log(trials.qu_kPa.to_numpy(float)) if k else np.zeros(0)
        if a.exclude_deposit:
            print("# note: the law's constants are fitted on the whole corpus; --exclude-deposit affects the in-context models only")
        mu, lo, hi = HierarchicalLaw().predict(Xt, yt, Xq, a.ll)
    else:
        df = load_corpus()
        if a.exclude_deposit:
            if a.exclude_deposit not in set(df.deposit):
                raise SystemExit(f"unknown deposit {a.exclude_deposit!r}; see: soilcement corpus")
            df = df[df.deposit != a.exclude_deposit]
        Xc, yc, _ = corpus_xy(df)
        if k:
            Xc = np.vstack([Xc, _X(trials, a)]); yc = np.concatenate([yc, np.log(trials.qu_kPa.to_numpy(float))])
        mu, lo, hi = predict_in_context(model, Xc, yc, Xq, device=a.device, checkpoint=a.checkpoint)
    return model, k, np.exp(mu), np.exp(lo), np.exp(hi)


def _note(k, model):
    kk = max(x for x in TYPICAL[model] if x <= k)
    m, md, w2 = TYPICAL[model][kk]
    adv = ("feasibility estimate only; run at least 3 trial mixes for a design" if k <= 1 else
           "preliminary design; 8 trial mixes are enough for design" if k < 8 else "design-level prediction")
    return (f"# model: {model}; trial mixes: {k}. Benchmark error with {kk} trial mixes on unseen deposits: "
            f"mean {m} %, median {md} %, {w2} % of strengths within a factor of two. {adv}.")


def cmd_predict(a):
    trials = _mixes(a.trials, True) if a.trials else None
    query = _mixes(a.query, False)
    model, k, q, lo, hi = _predict(a, trials, query)
    out = query.copy(); out["qu_pred_kPa"] = q.round(1); out["qu_p10_kPa"] = lo.round(1); out["qu_p90_kPa"] = hi.round(1)
    note = _note(k, model)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(note + "\n"); out.to_csv(fh, index=False)
        print(note); print(f"written {a.out}")
    else:
        print(note); print(out.to_string(index=False))


def cmd_design(a):
    trials = _mixes(a.trials, True) if a.trials else None
    grid = np.round(np.arange(a.min_binder, a.max_binder + 1e-9, a.step), 3)
    query = pd.DataFrame(dict(water_content_pct=a.water, binder_pct=grid, curing_days=a.curing, binder=a.binder,
                              pozzolan_pct_of_cement=a.pozzolan))
    query["equivalent_cement_pct"] = equivalent_cement(query.binder_pct, query.binder, query.pozzolan_pct_of_cement)
    model, k, q, lo, hi = _predict(a, trials, query)
    print(_note(k, model))
    print(f"# target qu = {a.target} kPa at water content {a.water} %, {a.curing} days, binder {a.binder}")
    med = grid[np.argmax(q >= a.target)] if np.any(q >= a.target) else None
    cau = grid[np.argmax(lo >= a.target)] if np.any(lo >= a.target) else None
    fmt = lambda v: f"{v:g} % of dry clay" if v is not None else f"not reached up to {a.max_binder:g} % (raise --max-binder or lower the water content)"
    print(f"binder content for predicted qu >= target: {fmt(med)}")
    print(f"binder content for 10th-percentile qu >= target (cautious): {fmt(cau)}")
    if a.table:
        print(pd.DataFrame(dict(binder_pct=grid, qu_pred_kPa=q.round(1), qu_p10_kPa=lo.round(1), qu_p90_kPa=hi.round(1))).to_string(index=False))
    print("# laboratory strength; apply the field-to-laboratory strength ratio of your design practice.")


def cmd_corpus(a):
    df = load_corpus()
    if a.export:
        df.to_csv(a.export, index=False); print(f"written {a.export}"); return
    print(f"{len(df)} mixes, {df.deposit.nunique()} deposits, {df.source_id.nunique()} sources, {df.laboratory.nunique()} laboratories, "
          f"{df.country.replace('', np.nan).nunique()} countries")
    for c, lab in (("LL_pct", "liquid limit (%)"), ("water_content_pct", "water content (%)"), ("equivalent_cement_pct", "equivalent cement (%)"),
                   ("curing_days", "curing (days)"), ("qu_kPa", "qu (kPa)")):
        v = df[c]; print(f"  {lab:24} 5th {v.quantile(.05):8.1f}  median {v.median():8.1f}  95th {v.quantile(.95):8.1f}")
    print(df.groupby("deposit").size().sort_values(ascending=False).to_string())


def main(argv=None):
    p = argparse.ArgumentParser(prog="soilcement", description="Strength of cement-admixed clay from Atterberg limits and a few trial mixes.")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def clay_args(s):
        s.add_argument("--ll", type=float, required=True, help="liquid limit of the project clay (%%)")
        s.add_argument("--pi", type=float, required=True, help="plasticity index (%%)")
        s.add_argument("--pl", type=float, default=None, help="plastic limit (%%); default LL - PI")
        s.add_argument("--trials", help="CSV of trial mixes with measured qu_kPa (omit for no trial mix)")
        s.add_argument("--model", default="auto", choices=["auto", "law", "tabicl", "tabicl-soil", "tabpfn3"],
                       help="auto: law with 0-1 trial mixes, TabICL from 2 (the paper's recommendation)")
        s.add_argument("--checkpoint", help="checkpoint for tabicl-soil (models/tabicl_soil.ckpt)")
        s.add_argument("--device", default="cpu", help="cpu or cuda")
        s.add_argument("--exclude-deposit", help="leave this corpus deposit out of the context (to test on a known clay)")

    s = sub.add_parser("predict", help="predict qu of untested mixes"); clay_args(s)
    s.add_argument("--query", required=True, help="CSV of mixes to predict"); s.add_argument("--out", help="output CSV")
    s.set_defaults(fn=cmd_predict)
    s = sub.add_parser("design", help="binder content needed for a target strength"); clay_args(s)
    s.add_argument("--target", type=float, required=True, help="target laboratory qu (kPa)")
    s.add_argument("--water", type=float, required=True, help="water content at mixing (%% of dry clay)")
    s.add_argument("--curing", type=float, default=28, help="curing time (days)")
    s.add_argument("--binder", default="opc", choices=["opc", "pcc", "slag"]); s.add_argument("--pozzolan", type=float, default=0.0, help="pozzolan, %% of cement")
    s.add_argument("--min-binder", type=float, default=5); s.add_argument("--max-binder", type=float, default=50); s.add_argument("--step", type=float, default=1)
    s.add_argument("--table", action="store_true", help="print the full strength-binder table")
    s.set_defaults(fn=cmd_design)
    s = sub.add_parser("corpus", help="summary or export of the corpus"); s.add_argument("--export"); s.set_defaults(fn=cmd_corpus)
    a = p.parse_args(argv); a.fn(a)


if __name__ == "__main__":
    main()
