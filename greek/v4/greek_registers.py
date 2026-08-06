"""
Register-sensitivity for the Greek archaizing measurement.

The manifest's `register` field is a scholarly judgment, and the measured cost
of archaizing depends on it: any Atticizing author left in the training set
teaches the model that late texts look early, which shrinks the very
displacement we are trying to measure.  Rather than defend one classification,
we run three and report all of them.

  A  as-received   the manifest's own labels
  B  strict        additionally treats as archaizing the authors whom the
                   literary-historical record independently describes as
                   classicizing -- the Neoplatonist biographical tradition,
                   the sophistic novelists, and Aelian, whom Philostratus
                   praises precisely for his Attic
  C  date-blind    no register labels at all: train only on texts composed
                   before the Second Sophistic begins (<= 50 CE), and treat
                   every later text as a test case.  This removes the
                   investigator's judgment from the training set entirely.

If the three agree in sign and rough magnitude, the conclusion does not rest
on the classification.
"""
import os, json, argparse
import numpy as np, pandas as pd
from scipy import stats
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("gmod", os.path.join(HERE, "greek_model.py"))
gmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(gmod)
Gram, LAM, wts, META = gmod.Gram, gmod.LAM, gmod.wts, gmod.META

# Authors described as classicizing/Atticizing in the standard literary
# histories, listed here independently of any model output.
RECLASS = [
    "aelian_various_history",        # Philostratus, VS 2.31, praises his Attic
    "plotinus_enneads",              # Neoplatonist literary prose
    "porphyry_life_plotinus",
    "iamblichus_pythagorean",
    "diogenes_laertius_lives",       # classicizing biographical compilation
    "appian_roman_history",          # moderately Atticizing historian
    "achilles_tatius",               # sophistic novel
    "chariton_callirhoe",            # sophistic novel
]


def run(D, train_mask, test_mask, label):
    Tr, Te = D[train_mask].copy(), D[test_mask].copy()
    feats = [c for c in D.columns if c not in META]
    Xa = Tr[feats].astype(float)
    keep = (Xa.std() > 0) & (Xa.isna().mean() < 0.2)
    feats = list(np.array(feats)[keep.values])
    med = Tr[feats].astype(float).median()
    X = Tr[feats].astype(float).fillna(med).values
    Xt = Te[feats].astype(float).fillna(med).values
    y = Tr.date_ce.values.astype(float); g = Tr.unit.values; gt = Te.unit.values
    texts = list(pd.unique(g)); tdate = {t: y[g == t][0] for t in texts}

    tt, pp = [], []
    for t in texts:
        te = g == t; tr = ~te
        inner = [x for x in texts if x != t]
        wb = dict(zip(inner, wts([tdate[x] for x in inner])))
        wv = np.array([wb[u] for u in g[tr]])
        gr = Gram(X[tr], y[tr], wv); gin = g[tr]
        err = {l: [] for l in LAM}
        for bb in inner:
            m = gin == bb
            pr = gr.solve(X[tr][m], drop=m)
            for l in LAM: err[l].append(abs(np.median(pr[l]) - tdate[bb]))
        bl = min(LAM, key=lambda l: np.mean(err[l]))
        tt.append(tdate[t]); pp.append(float(np.median(gr.solve(X[te])[bl])))
    tt, pp = np.array(tt), np.array(pp)
    S = float(np.clip(tt.std() / pp.std(), 0.5, 8.0))
    cal = tt.mean() + S * (pp - pp.mean())
    resid = tt - cal
    mae = float(np.abs(resid).mean())
    base = float(np.abs(tt - tt.mean()).mean())
    rho = float(stats.spearmanr(tt, cal)[0])

    wb = dict(zip(texts, wts([tdate[x] for x in texts])))
    wv = np.array([wb[u] for u in g])
    gr = Gram(X, y, wv)
    err = {l: [] for l in LAM}
    for t_ in texts:
        m = g == t_
        pr = gr.solve(X[m], drop=m)
        for l in LAM: err[l].append(abs(np.median(pr[l]) - tdate[t_]))
    bl = min(LAM, key=lambda l: np.mean(err[l]))
    predt = tt.mean() + S * (gr.solve(Xt)[bl] - pp.mean())

    rows = []
    for u in pd.unique(gt):
        s = predt[gt == u]
        tru = float(Te.date_ce[gt == u].iloc[0]); pv = float(np.median(s))
        rows.append(dict(text=u, author=Te.author[gt == u].iloc[0],
                         truth=int(tru), pred=round(pv), shift=round(pv - tru),
                         n_chunks=int((gt == u).sum())))
    A = pd.DataFrame(rows).sort_values("truth")
    out = dict(variant=label, n_train=len(texts), n_train_chunks=int(len(Tr)),
               n_test=int(Te.unit.nunique()), mae=mae, mae_baseline=base,
               rho=rho, mean_shift=float(A["shift"].mean()),
               median_shift=float(A["shift"].median()),
               n_early=int((A["shift"] < 0).sum()))
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    print(f"  train {len(texts)} texts / {len(Tr)} passages | test {out['n_test']} texts")
    print(f"  LOTO MAE {mae:.0f} yr (baseline {base:.0f})  rho {rho:+.3f}")
    print(f"{'author':<26}{'true':>6}{'pred':>7}{'shift':>8}")
    for _, r in A.iterrows():
        print(f"{r.author[:25]:<26}{r.truth:6d}{r.pred:7d}{r['shift']:+8d}")
    print(f"  mean displacement {out['mean_shift']:+.0f} yr | "
          f"{out['n_early']}/{len(A)} dated too early")
    return out, A


def main():
    D = pd.read_csv(os.path.join(HERE, "greek_chunks_500.csv"))
    D = D[(D.n_tok >= 100) & (D.register != "LXX")].copy()
    res = []

    att = D.register == "Atticizing"
    o, A = run(D, ~att, att, "A  as-received manifest labels")
    res.append(o); A.to_csv(f"{HERE}/greek_att_A.csv", index=False)

    att2 = att | D.unit.isin(RECLASS)
    o, A = run(D, ~att2, att2, "B  strict: classicizing authors reclassified")
    res.append(o); A.to_csv(f"{HERE}/greek_att_B.csv", index=False)

    early = D.date_ce <= 50
    o, A = run(D, early, ~early & att, "C  date-blind: train only on <= 50 CE")
    res.append(o); A.to_csv(f"{HERE}/greek_att_C.csv", index=False)

    R = pd.DataFrame(res)
    R.to_csv(f"{HERE}/greek_register_variants.csv", index=False)
    json.dump(res, open(f"{HERE}/greek_register_variants.json", "w"), indent=2)
    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    print(R[["variant", "n_train", "n_test", "mae", "rho",
             "mean_shift", "n_early"]].to_string(index=False))


if __name__ == "__main__":
    main()
