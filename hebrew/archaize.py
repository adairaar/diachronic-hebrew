"""
Synthetic archaizing: how many years of apparent antiquity can substitution buy?

Take a securely dated LATE book, replace its Late-Biblical-Hebrew forms with
their Classical counterparts at a controlled rate, re-extract features, and
date it with a model trained on the other 24 books (unmodified).  The apparent
date shift as a function of substitution rate is a calibration curve for
archaizing.

This is the positive control the resistant-model diagnostic never had: the
ground truth is known, because we performed the archaizing ourselves.

Substitutions are the classic CBH/LBH diagnostics, applied LBH -> CBH:
    C     -> >CR     she -> asher        (relative)
    >NJ   -> >NKJ    ani -> anochi       (1sg pronoun)
    MLKWT/-> MMLKH/  malkut -> mamlakah  (kingdom)
    >JN/  -> L>      ein -> lo           (negation; also changes sp subs->nega)

Each is a lexeme swap with matching part of speech except the last, where the
part of speech is changed too, as a real substitution would.
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import numpy as np, pandas as pd, collections, importlib.util, argparse, json

spec = importlib.util.spec_from_file_location("big", DH.script("chunk_extract_big.py"))
big = importlib.util.module_from_spec(spec); spec.loader.exec_module(big)
pt = importlib.util.spec_from_file_location("pt", DH.script("predict_targets.py"))
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

SUBS = {"C": (">CR", None), ">NJ": (">NKJ", None),
        "MLKWT/": ("MMLKH/", None), ">JN/": ("L>", "nega")}
LATE = {"Chronicles": 330, "Esther": 300, "Ecclesiastes": 250,
        "Daniel": 167, "Ezra": 380, "Nehemiah": 380}
BHSA = {"Ezra": "Esra", "Nehemiah": "Nehemia", "Ecclesiastes": "Ecclesiastes",
        "Esther": "Esther", "Daniel": "Daniel"}
RANGED = {"Chronicles": [("Chronica_I", [(1, 29)]), ("Chronica_II", [(1, 36)])],
          "Daniel": [("Daniel", [(1, 1), (8, 12)])]}


class ProxyF:
    """Wraps F so lex.v / sp.v return substituted values for chosen nodes."""
    def __init__(self, F, lexmap, spmap):
        self._F = F
        class _A:
            def __init__(s, real, m): s.real, s.m = real, m
            def v(s, n): return s.m.get(n, s.real.v(n))
        self.lex = _A(F.lex, lexmap)
        self.sp = _A(F.sp, spmap)
    def __getattr__(self, k): return getattr(self._F, k)


class ProxyAPI:
    def __init__(self, api, F): self.F, self.L, self.T = F, api.L, api.T


def unit_words(api, uid):
    spec_ = RANGED.get(uid) or [(BHSA.get(uid, uid), [(1, 999)])]
    return big.unit_words(api, spec_)


def chunks_of(api, words, target=500):
    L = api.L
    verses, cur, curv = [], [], None
    for w in words:
        v = L.u(w, "verse"); v = v[0] if v else None
        if v != curv and cur: verses.append(cur); cur = []
        curv = v; cur.append(w)
    if cur: verses.append(cur)
    out, buf = [], []
    for v in verses:
        buf.extend(v)
        if len(buf) >= target: out.append(buf); buf = []
    if buf:
        if out and len(buf) < target * .5: out[-1].extend(buf)
        else: out.append(buf)
    return out


def main(rates, seed=0):
    api = big.load(); F = api.F
    rng = np.random.default_rng(seed)
    allw = [w for w in F.otype.s("word") if F.language.v(w) == "Hebrew"]
    tl = [k for k, _ in collections.Counter(F.lex.v(w) for w in allw).most_common(250)]
    tp = [k for k, _ in collections.Counter(F.sp.v(w) for w in allw).most_common(14)]
    ph = list(F.otype.s("phrase")); cl = list(F.otype.s("clause"))
    tt = [k for k, _ in collections.Counter(F.typ.v(p) for p in ph).most_common(12)]
    tf = [k for k, _ in collections.Counter(F.function.v(p) for p in ph).most_common(16)]
    trl = [k for k, _ in collections.Counter(F.rela.v(c) for c in cl).most_common(12)]
    tct = [k for k, _ in collections.Counter(F.typ.v(c) for c in cl).most_common(14)]

    # ── dated corpus, unmodified: model + calibration ────────────────────────
    Dd = pd.read_csv(DH.f("big_features_500.csv"))
    y = Dd.date_bce.values.astype(float); g = Dd.unit.values
    books = list(pd.unique(g)); bdate = {b: y[g == b][0] for b in books}

    rows = []
    for uid, truth in LATE.items():
        words = unit_words(api, uid)
        elig = {lx: [w for w in words if F.lex.v(w) == lx] for lx in SUBS}
        n_el = {k: len(v) for k, v in elig.items()}
        for r in rates:
            lexmap, spmap = {}, {}
            n_sub = 0
            for lx, (tgt, newsp) in SUBS.items():
                pool = elig[lx]
                if not pool: continue
                k = int(round(r * len(pool)))
                for w in rng.choice(pool, size=k, replace=False) if k else []:
                    lexmap[int(w)] = tgt
                    if newsp: spmap[int(w)] = newsp
                n_sub += k
            papi = ProxyAPI(api, ProxyF(F, lexmap, spmap))
            recs = []
            for i, ch in enumerate(chunks_of(api, words)):
                fe = big.chunk_feats(ch, papi, tl, tp, tt, tf, trl, tct)
                recs.append(dict(chunk_id=f"{uid}_{r}_{i}", unit=uid, date_bce=np.nan,
                                 genre=None, register=None, n_words=len(ch), **fe))
            M = pd.DataFrame(recs)

            feats = [c for c in Dd.columns if c not in PT.META and c in M.columns]
            Xa = Dd[feats].astype(float)
            keep = (Xa.std() > 0) & (Xa.isna().mean() < .2)
            fl = list(np.array(feats)[keep.values])
            med = Dd[fl].astype(float).median()
            Xd = Dd[fl].astype(float).fillna(med).values
            Xm = M[fl].astype(float).fillna(med).values

            tr = g != uid                       # train on the other 24 books
            trb = [b for b in books if b != uid]
            wb = dict(zip(trb, PT.wts([bdate[b] for b in trb])))
            wv = np.array([wb[u] for u in g[tr]])
            best, bl = np.inf, PT.LAM[0]
            for lam in PT.LAM:
                e = []
                for bb in trb:
                    m = g[tr] != bb
                    pr = PT.fit_predict(Xd[tr][m], y[tr][m], wv[m], Xd[tr][~m], lam)
                    e.append(abs(np.median(pr) - bdate[bb]))
                if np.mean(e) < best: best, bl = np.mean(e), lam
            ot, op = [], []
            for bb in trb:
                m = g[tr] != bb
                pr = PT.fit_predict(Xd[tr][m], y[tr][m], wv[m], Xd[tr][~m], bl)
                ot.append(bdate[bb]); op.append(float(np.median(pr)))
            ot, op = np.array(ot), np.array(op)
            S = float(np.clip(ot.std() / op.std(), .5, 8.))
            pred = ot.mean() + S * (float(np.median(
                PT.fit_predict(Xd[tr], y[tr], wv, Xm, bl))) - op.mean())
            rows.append(dict(unit=uid, truth=truth, rate=r, n_sub=n_sub,
                             n_words=len(words), pred=pred))
            print(f"  {uid:<14} rate={r:.2f}  {n_sub:4d} tokens swapped  "
                  f"-> {pred:6.0f} BCE", flush=True)
        print(f"    eligible tokens: {n_el}", flush=True)

    R = pd.DataFrame(rows)
    R.to_csv(DH.f("archaize_results.csv"), index=False)
    print("\n" + "=" * 72)
    print("APPARENT ANTIQUITY PURCHASED BY FULL LEXICAL ARCHAIZING")
    print("=" * 72)
    print(f"{'unit':<14}{'true':>6}{'r=0':>8}{'r=1':>8}{'shift':>8}{'swaps':>7}{'per 1k w':>10}")
    for uid in LATE:
        s = R[R.unit == uid]
        if s.empty: continue
        a = s[s.rate == 0].pred.iloc[0]; b = s[s.rate == 1.0].pred.iloc[0]
        ns = s[s.rate == 1.0].n_sub.iloc[0]; nw = s.n_words.iloc[0]
        print(f"{uid:<14}{LATE[uid]:6d}{a:8.0f}{b:8.0f}{b-a:+8.0f}{ns:7d}"
              f"{ns/nw*1000:10.1f}")
    tot = [(R[(R.unit == u) & (R.rate == 1.0)].pred.iloc[0]
            - R[(R.unit == u) & (R.rate == 0)].pred.iloc[0]) for u in LATE
           if not R[R.unit == u].empty]
    print(f"\n  mean apparent shift from 100% lexical archaizing: {np.mean(tot):+.0f} yr")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=float, nargs="+", default=[0, .25, .5, .75, 1.0])
    main(ap.parse_args().rates)
