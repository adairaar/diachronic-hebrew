"""
Expanded feature extraction at chunk level.

Feature families (all computed per chunk, rates per 1000 words unless noted):
  LEX   top-K lexeme rates
  POS   part-of-speech unigram + bigram rates
  VERB  verb tense, stem, and tense x stem cross rates
  AGR   person / number / gender / state / pronominal-suffix rates
  PHR   phrase type and phrase function rates
  CLS   clause relation rates, clause-type bigrams
  STRUCT  mean phrase length, clause length, clauses per sentence, etc.

Everything is derived from BHSA node features, so nothing here depends on the
hand-picked 64-feature set the earlier work used.
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import os, json, argparse, collections
import numpy as np, pandas as pd
from tf.fabric import Fabric

TF_PATH = os.environ.get("BHSA_TF") or os.path.expanduser(
    "~/text-fabric-data/github/ETCBC/bhsa/tf/2021")

BHSA_NAME = {"Numbers":"Numeri","Deuteronomy":"Deuteronomium","Lamentations":"Threni",
             "Ezekiel":"Ezechiel","Obadiah":"Obadia","Jonah":"Jona","Malachi":"Maleachi",
             "Zephaniah":"Zephania","Habakkuk":"Habakuk","Micah":"Micha","Isaiah":"Jesaia",
             "Jeremiah":"Jeremia","Zechariah":"Sacharia","Ezra":"Esra","Nehemiah":"Nehemia",
             "1_Chronicles":"Chronica_I","2_Chronicles":"Chronica_II","Judges":"Judices"}
RANGED = {
    "Isaiah_1":[("Jesaia",[(1,39)])], "Isaiah_2":[("Jesaia",[(40,55)])],
    "Isaiah_3":[("Jesaia",[(56,66)])], "Zechariah_1":[("Sacharia",[(1,8)])],
    "Zechariah_2":[("Sacharia",[(9,14)])],
    "Chronicles":[("Chronica_I",[(1,29)]),("Chronica_II",[(1,36)])],
    "Daniel":[("Daniel",[(1,1),(8,12)])],
    "Jer_oracle":[("Jeremia",[(1,6),(8,10),(12,16),(19,20),(22,23),(30,31),(46,51)])],
}
WHOLE = ["Amos","Hosea","Micah","Zephaniah","Nahum","Habakkuk","Lamentations",
         "Ezekiel","Obadiah","Joel","Jonah","Malachi","Haggai","Ezra","Nehemiah",
         "Esther","Ecclesiastes"]


def load():
    TF = Fabric(locations=[TF_PATH], silent="deep")
    return TF.load("book chapter verse lex sp pdp vt vs ps nu gn prs st typ "
                   "function rela language", silent="deep")


def unit_words(api, spec):
    """Words in the given chapter ranges, restricted to Hebrew.

    The restriction is not cosmetic.  Ezra 4:8-6:18 and 7:12-26 are Aramaic, and
    without this filter they made up 28.9% of the Ezra unit.  Daniel's Aramaic
    (2:4b-7:28) is excluded by its chapter range instead, and Jer 10:11 adds a
    further 19 words.  A model of Hebrew morphosyntax must not be fitted to
    Aramaic.
    """
    F, L, T = api.F, api.L, api.T
    ws = []
    for book, ranges in spec:
        bn = T.nodeFromSection((book,))
        if bn is None:
            print(f"   !! book not found: {book}"); continue
        for ch in L.d(bn, "chapter"):
            c = int(F.chapter.v(ch))
            if any(s <= c <= e for s, e in ranges):
                ws.extend(w for w in L.d(ch, "word")
                          if F.language.v(w) == "Hebrew")
    return ws


def chunk_feats(words, api, top_lex, top_pos, top_typ, top_fun, top_rela,
                top_ctyp=None):
    F, L = api.F, api.L
    n = len(words)
    f = {}
    R = lambda c: c / n * 1000.0

    lex = [F.lex.v(w) for w in words]
    sp  = [F.sp.v(w) for w in words]
    pdp = [F.pdp.v(w) for w in words]

    lc = collections.Counter(lex)
    for k in top_lex:
        f[f"lex_{k}"] = R(lc.get(k, 0))

    pc = collections.Counter(sp)
    for k in top_pos:
        f[f"pos_{k}"] = R(pc.get(k, 0))
    bg = collections.Counter(zip(sp, sp[1:]))
    for a in top_pos:
        for b in top_pos:
            f[f"pb_{a}_{b}"] = R(bg.get((a, b), 0))

    pdc = collections.Counter(pdp)
    for k in top_pos:
        f[f"pdp_{k}"] = R(pdc.get(k, 0))

    vb = [w for w in words if F.sp.v(w) == "verb"]
    nv = max(len(vb), 1)
    vt = collections.Counter(F.vt.v(w) for w in vb)
    vs = collections.Counter(F.vs.v(w) for w in vb)
    vx = collections.Counter((F.vt.v(w), F.vs.v(w)) for w in vb)
    f["verb_rate"] = R(len(vb))
    # BHSA vt values are perf/impf/wayq/impv/infc/infa/ptca/ptcp; there is no
    # "weqt" -- weqatal is encoded at clause level as the WQt0 clause type.
    for k in ["perf","impf","wayq","impv","infc","infa","ptca","ptcp"]:
        f[f"vt_{k}"] = R(vt.get(k, 0)); f[f"vtf_{k}"] = vt.get(k, 0) / nv
    for k in ["qal","hif","piel","nif","pual","hit","hof","hsht","poal","poel"]:
        f[f"vs_{k}"] = R(vs.get(k, 0)); f[f"vsf_{k}"] = vs.get(k, 0) / nv
    for a in ["perf","impf","wayq","impv","ptca"]:
        for b in ["qal","hif","piel","nif"]:
            f[f"vx_{a}_{b}"] = vx.get((a, b), 0) / nv

    for feat, vals in [("ps", ["p1","p2","p3"]), ("nu", ["sg","pl","du"]),
                       ("gn", ["m","f"]), ("st", ["a","c","e"])]:
        c = collections.Counter(getattr(F, feat).v(w) for w in words)
        for v in vals:
            f[f"{feat}_{v}"] = R(c.get(v, 0))
    prs = collections.Counter(F.prs.v(w) for w in words)
    f["prs_any"] = R(sum(v for k, v in prs.items() if k not in ("absent", "n/a", None)))
    for k in ["W","J","M","K","H","HM","NW","MW"]:
        f[f"prs_{k}"] = R(prs.get(k, 0))

    phrases = sorted({p for w in words for p in L.u(w, "phrase")})
    clauses = sorted({c for w in words for c in L.u(w, "clause")})
    sents   = sorted({s for w in words for s in L.u(w, "sentence")})
    tc = collections.Counter(F.typ.v(p) for p in phrases)
    fc = collections.Counter(F.function.v(p) for p in phrases)
    rc = collections.Counter(F.rela.v(c) for c in clauses)
    npz = max(len(phrases), 1); ncl = max(len(clauses), 1)
    for k in top_typ: f[f"typ_{k}"] = tc.get(k, 0) / npz
    for k in top_fun: f[f"fun_{k}"] = fc.get(k, 0) / npz
    for k in top_rela: f[f"rela_{k}"] = rc.get(k, 0) / ncl
    # Clause TYPE carries the Hebrew verbal-system contrast (Way0 narrative
    # wayyiqtol, WQt0 weqatal, xQt0 fronted qatal, NmCl verbless) and is distinct
    # from clause RELA.  Earlier versions counted clause types but indexed them
    # with relation keys, whose value sets are disjoint, so every one of these
    # features was structurally zero.
    ctyp = [F.typ.v(c) for c in clauses]
    ctc = collections.Counter(ctyp)
    tcl = top_ctyp or []
    for k in tcl:
        f[f"ctyp_{k}"] = ctc.get(k, 0) / ncl
    cbg = collections.Counter(zip(ctyp, ctyp[1:]))
    for a in tcl[:6]:
        for b in tcl[:6]:
            f[f"cb_{a}_{b}"] = cbg.get((a, b), 0) / ncl

    f["phrase_len"]   = n / npz
    f["clause_len"]   = n / ncl
    f["cl_per_sent"]  = ncl / max(len(sents), 1)
    f["ph_per_clause"] = npz / ncl
    f["sent_len"]     = n / max(len(sents), 1)
    return f


def main(target, k_lex):
    api = load(); F, L = api.F, api.L
    allw = [w for w in F.otype.s("word") if F.language.v(w) == "Hebrew"]
    top_lex  = [k for k, _ in collections.Counter(F.lex.v(w) for w in allw).most_common(k_lex)]
    top_pos  = [k for k, _ in collections.Counter(F.sp.v(w) for w in allw).most_common(14)]
    ph = list(F.otype.s("phrase")); cl = list(F.otype.s("clause"))
    top_typ  = [k for k, _ in collections.Counter(F.typ.v(p) for p in ph).most_common(12)]
    top_fun  = [k for k, _ in collections.Counter(F.function.v(p) for p in ph).most_common(16)]
    top_rela = [k for k, _ in collections.Counter(F.rela.v(c) for c in cl).most_common(12)]
    top_ctyp = [k for k, _ in collections.Counter(F.typ.v(c) for c in cl).most_common(14)]
    print(f"vocab: {len(top_lex)} lex, {len(top_pos)} pos, {len(top_typ)} typ, "
          f"{len(top_fun)} fun, {len(top_rela)} rela, {len(top_ctyp)} ctyp")

    man = json.load(open(DH.f("corpus_manifest_v2.json")))
    rows = []
    for grp in ("training", "holdouts"):
        for t in man[grp]:
            uid = t["id"]
            spec = RANGED.get(uid) or ([(BHSA_NAME.get(uid, uid), [(1, 999)])]
                                       if uid in WHOLE else None)
            if spec is None:
                print(f"  skip {uid}"); continue
            words = unit_words(api, spec)
            if not words: continue
            verses, cur, curv = [], [], None
            for w in words:
                v = L.u(w, "verse"); v = v[0] if v else None
                if v != curv and cur: verses.append(cur); cur = []
                curv = v; cur.append(w)
            if cur: verses.append(cur)
            chunks, buf = [], []
            for v in verses:
                buf.extend(v)
                if len(buf) >= target: chunks.append(buf); buf = []
            if buf:
                if chunks and len(buf) < target * 0.5: chunks[-1].extend(buf)
                else: chunks.append(buf)
            for i, ch in enumerate(chunks):
                fe = chunk_feats(ch, api, top_lex, top_pos, top_typ, top_fun, top_rela,
                             top_ctyp)
                rows.append(dict(chunk_id=f"{uid}_c{i:03d}", unit=uid,
                                 date_bce=t["date_bce"], genre=t.get("genre"),
                                 register=t.get("register"), n_words=len(ch), **fe))
            print(f"  {uid:<15} {len(words):6d} w -> {len(chunks):3d} chunks")
    D = pd.DataFrame(rows)
    META = ("chunk_id", "unit", "date_bce", "genre", "register", "n_words")

    # Two matrices are written, and which is which is a decision rather than an
    # accident of when the extractor was last run.
    #
    # The clause-type family encodes the Hebrew verbal system (Way0 narrative
    # wayyiqtol, WQt0 weqatal, xQt0 fronted qatal, NmCl verbless).  Its members
    # are individually among the most date-correlated features in the corpus,
    # and adding them nonetheless leaves out-of-sample performance unchanged,
    # because their information is already carried by the verb-morphology and
    # part-of-speech transition features (see quicklobo.py).  The reported model
    # therefore excludes them, and the file it reads excludes them too, so that
    # a fresh run of this script reproduces the published feature set rather
    # than a superset of it.
    ctyp_cols = [c for c in D.columns
                 if c.startswith("ctyp_") or c.startswith("cb_")]
    base = DH.f(f"big_features_{target}")

    D.drop(columns=ctyp_cols).to_csv(f"{base}.csv", index=False)
    D.to_csv(f"{base}_ctyp.csv", index=False)

    n_all = len([c for c in D.columns if c not in META])
    print(f"\n{len(D)} chunks")
    print(f"  {n_all - len(ctyp_cols):>4} features -> {base}.csv        "
          f"(reported model)")
    print(f"  {n_all:>4} features -> {base}_ctyp.csv   "
          f"(+{len(ctyp_cols)} clause-type, for the comparison only)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--klex", type=int, default=250)
    a = ap.parse_args()
    main(a.target, a.klex)
