"""
Re-extract the three poems at VERSE precision and re-date them.

The project's unit specs defined the poems as whole chapters, which pulls in
prose that is not part of the poem: 47% of "Song of the Sea" and 26% of "Song
of Moses" by word count.  Narrative is biased +143 yr late under this model, so
prose contamination pushes a poem's estimate later.

Also splits the Song of Moses so the archaic divine-council section (32:8-9,
Elyon apportioning the nations) sits in its own block.
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import numpy as np, pandas as pd, collections, importlib.util
from scipy import stats

spec = importlib.util.spec_from_file_location("big", DH.script("chunk_extract_big.py"))
big = importlib.util.module_from_spec(spec); spec.loader.exec_module(big)
pt = importlib.util.spec_from_file_location("pt", DH.script("predict_targets.py"))
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

UNITS = {
    "SongSea_poem":      ("Exodus", 15, 1, 18),
    "SongSea_chapter":   ("Exodus", 15, 1, 99),
    "SongSea_prose":     ("Exodus", 15, 19, 27),
    "SongMoses_poem":    ("Deuteronomium", 32, 1, 43),
    "SongMoses_chapter": ("Deuteronomium", 32, 1, 99),
    "SongMoses_prose":   ("Deuteronomium", 32, 44, 52),
    "SongMoses_1_25":    ("Deuteronomium", 32, 1, 25),
    "SongMoses_26_43":   ("Deuteronomium", 32, 26, 43),
    "SongDeborah_poem":  ("Judices", 5, 2, 31),
    "SongDeborah_chap":  ("Judices", 5, 1, 99),
}


def verse_words(api, book, ch, v1, v2):
    F, L, T = api.F, api.L, api.T
    bn = T.nodeFromSection((book, ch))
    if bn is None: return []
    out = []
    for vs in L.d(bn, "verse"):
        if v1 <= int(F.verse.v(vs)) <= v2:
            out.extend(L.d(vs, "word"))
    return out


def main():
    api = big.load(); F = api.F
    allw = [w for w in F.otype.s("word") if F.language.v(w) == "Hebrew"]
    tl = [k for k, _ in collections.Counter(F.lex.v(w) for w in allw).most_common(250)]
    tp = [k for k, _ in collections.Counter(F.sp.v(w) for w in allw).most_common(14)]
    ph = list(F.otype.s("phrase")); cl = list(F.otype.s("clause"))
    tt = [k for k, _ in collections.Counter(F.typ.v(p) for p in ph).most_common(12)]
    tf = [k for k, _ in collections.Counter(F.function.v(p) for p in ph).most_common(16)]
    tr = [k for k, _ in collections.Counter(F.rela.v(c) for c in cl).most_common(12)]
    tct = [k for k, _ in collections.Counter(F.typ.v(c) for c in cl).most_common(14)]

    rows = []
    for uid, (bk, ch, v1, v2) in UNITS.items():
        w = verse_words(api, bk, ch, v1, v2)
        if not w: print(f"  !! {uid}"); continue
        fe = big.chunk_feats(w, api, tl, tp, tt, tf, tr, tct)
        rows.append(dict(chunk_id=uid, unit=uid, date_bce=np.nan, genre=None,
                         register=None, n_words=len(w), **fe))
        print(f"  {uid:<20} {len(w):4d} words")
    P = pd.DataFrame(rows)
    P.to_csv(DH.f("poem_chunks.csv"), index=False)

    # ── fit the recommended model on the dated corpus, predict the poems ──────
    Dd = pd.read_csv(DH.f("big_features_500.csv"))
    feats = [c for c in Dd.columns if c not in PT.META and c in P.columns]
    Xd_all = Dd[feats].astype(float)
    keep = (Xd_all.std() > 0) & (Xd_all.isna().mean() < 0.2)
    feats = list(np.array(feats)[keep.values])
    med = Dd[feats].astype(float).median()
    Xd = Dd[feats].astype(float).fillna(med).values
    Xp = P[feats].astype(float).fillna(med).values
    y = Dd.date_bce.values.astype(float); g = Dd.unit.values
    books = list(pd.unique(g)); bdate = {b: y[g == b][0] for b in books}

    oof_t, oof_p = [], []
    for b in books:
        te = g == b; tr_ = ~te
        inner = [x for x in books if x != b]
        wb = dict(zip(inner, PT.wts([bdate[x] for x in inner])))
        wv = np.array([wb[u] for u in g[tr_]])
        best, blam = np.inf, PT.LAM[0]
        for lam in PT.LAM:
            e = []
            for bb in inner:
                m = g[tr_] != bb
                pr = PT.fit_predict(Xd[tr_][m], y[tr_][m], wv[m], Xd[tr_][~m], lam)
                e.append(abs(np.median(pr) - y[tr_][~m][0]))
            if np.mean(e) < best: best, blam = np.mean(e), lam
        pr = PT.fit_predict(Xd[tr_], y[tr_], wv, Xd[te], blam)
        oof_t.append(bdate[b]); oof_p.append(float(np.median(pr)))
    oof_t, oof_p = np.array(oof_t), np.array(oof_p)
    S = float(np.clip(oof_t.std() / oof_p.std(), .5, 8.))
    cal = lambda v: oof_t.mean() + S * (v - oof_p.mean())
    resid = oof_t - cal(oof_p)
    ar = np.sort(np.abs(resid)); n = len(ar)
    q68 = ar[min(int(np.ceil((n+1)*.68))-1, n-1)]

    wb = dict(zip(books, PT.wts([bdate[x] for x in books])))
    wv = np.array([wb[u] for u in g])
    best, blam = np.inf, PT.LAM[0]
    for lam in PT.LAM:
        e = []
        for b in books:
            m = g != b
            pr = PT.fit_predict(Xd[m], y[m], wv[m], Xd[~m], lam)
            e.append(abs(np.median(pr) - bdate[b]))
        if np.mean(e) < best: best, blam = np.mean(e), lam
    pred = cal(PT.fit_predict(Xd, y, wv, Xp, blam))

    print(f"\nconformal 68% half-width +/-{q68:.0f} yr\n")
    print(f"{'unit':<22}{'words':>7}{'date':>7}{'68% interval':>20}{'P(post-exilic)':>16}")
    print("-" * 74)
    out = []
    for i, uid in enumerate(P.unit):
        p_ = float(pred[i]); pp = float(np.mean((p_ + resid) < 586))
        out.append(dict(unit=uid, n_words=int(P.n_words[i]), pred=p_,
                        lo68=p_-q68, hi68=p_+q68, p_post=pp))
        print(f"{uid:<22}{P.n_words[i]:7d}{p_:7.0f}"
              f"{f'  {p_+q68:.0f} - {p_-q68:.0f} BCE':>20}{pp:16.2f}")
    pd.DataFrame(out).to_csv(DH.f("poem_predictions.csv"), index=False)


if __name__ == "__main__":
    main()
