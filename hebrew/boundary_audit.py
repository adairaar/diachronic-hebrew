"""
Audit the chunk boundary logic.

Phrase, clause and sentence features are collected with L.u(w, level) over the
words of a chunk, which returns every object *overlapping* the chunk.  An object
straddling a boundary is therefore counted in both adjacent chunks, and its type
is credited to both as though it lay wholly inside each.  The counts npz and ncl
are also the denominators for the typ_/fun_/rela_ rates and for phrase_len,
clause_len, sent_len, cl_per_sent and ph_per_clause.

This measures how much that matters.  Chunks are cut at verse boundaries, so the
question is how often a phrase, clause or sentence spans a verse boundary.
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import os, json, collections
import numpy as np, pandas as pd
from tf.fabric import Fabric

TF_PATH = os.environ.get("BHSA_TF") or os.path.expanduser(
    "~/text-fabric-data/github/ETCBC/bhsa/tf/2021")
api = Fabric(locations=[TF_PATH], silent="deep").load(
    "book chapter verse language typ rela otype", silent="deep")
F, L, T = api.F, api.L, api.T

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


def unit_words(spec):
    ws = []
    for book, ranges in spec:
        bn = T.nodeFromSection((book,))
        if bn is None: continue
        for ch in L.d(bn, "chapter"):
            if any(s <= int(F.chapter.v(ch)) <= e for s, e in ranges):
                ws.extend(w for w in L.d(ch, "word") if F.language.v(w) == "Hebrew")
    return ws


def chunks_of(words, target=500):
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
        if out and len(buf) < target * 0.5: out[-1].extend(buf)
        else: out.append(buf)
    return out


rows = []
for uid in list(RANGED) + WHOLE:
    spec = RANGED.get(uid) or [(BHSA_NAME.get(uid, uid), [(1, 999)])]
    words = unit_words(spec)
    if not words: continue
    for ci, ch in enumerate(chunks_of(words)):
        S = set(ch)
        r = dict(unit=uid, chunk=ci, n=len(ch))
        for lvl in ("phrase", "clause", "sentence"):
            objs = sorted({o for w in ch for o in L.u(w, lvl)})
            full = [o for o in objs if set(L.d(o, "word")) <= S]
            r[f"{lvl}_overlap"] = len(objs)
            r[f"{lvl}_full"] = len(full)
            r[f"{lvl}_partial"] = len(objs) - len(full)
        rows.append(r)

D = pd.DataFrame(rows)
D.to_csv(DH.f("boundary_audit.csv"), index=False)

print(f"{len(D)} chunks from {D.unit.nunique()} units\n")
print(f"{'level':<10}{'objects':>10}{'partial':>10}{'% partial':>11}"
      f"{'inflation':>11}")
out = {}
for lvl in ("phrase", "clause", "sentence"):
    ov = D[f"{lvl}_overlap"].sum(); pa = D[f"{lvl}_partial"].sum()
    infl = ov / D[f"{lvl}_full"].sum()
    out[lvl] = dict(overlap=int(ov), partial=int(pa), pct=100 * pa / ov,
                    inflation=float(infl))
    print(f"{lvl:<10}{ov:>10}{pa:>10}{100*pa/ov:>10.1f}%{infl:>10.3f}x")

print("\nper-chunk effect on the derived length features:")
for num, den, name in [("n", "phrase", "phrase_len"), ("n", "clause", "clause_len"),
                       ("n", "sentence", "sent_len")]:
    a = (D[num] / D[f"{den}_overlap"]).mean()
    b = (D[num] / D[f"{den}_full"].replace(0, np.nan)).mean()
    print(f"  {name:<12} as computed {a:6.2f}   fully-contained only {b:6.2f}   "
          f"bias {100*(a-b)/b:+5.1f}%")

worst = D.assign(pct=100 * D.clause_partial / D.clause_overlap).nlargest(5, "pct")
print("\nchunks with the most partial clauses:")
for _, r in worst.iterrows():
    print(f"  {r.unit:<14} chunk {int(r.chunk):>3}  n={int(r.n):>4}  "
          f"{int(r.clause_partial)}/{int(r.clause_overlap)} clauses partial")
json.dump(out, open(DH.f("boundary_audit.json"), "w"), indent=2)
print("\nwrote boundary_audit.csv, boundary_audit.json")
