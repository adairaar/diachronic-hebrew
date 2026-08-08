"""
Chunk-level feature extraction from BHSA.

Splits every dated unit into contiguous chunks of ~TARGET words, respecting
verse boundaries, and computes the SAME feature set as the book-level
pipeline -- guaranteed identical, because it calls the existing extract_unit()
with its word-collection function patched to return the chunk.

Output: chunk_features_<TARGET>.csv, one row per chunk.
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import os, sys, json, importlib.util, argparse
import numpy as np, pandas as pd

TF_PATH = os.environ.get("BHSA_TF") or os.path.expanduser(
    "~/text-fabric-data/github/ETCBC/bhsa/tf/2021")

spec = importlib.util.spec_from_file_location(
    "ex", os.path.join(DH.HEBREW, "hierarchical_bayes", "00_extract_features.py"))
ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
ORIG_WFR = ex.words_for_ranges       # keep the real one; the patch below is scoped

WHOLE = {k: k for k in [
    "Amos", "Hosea", "Micah", "Zephaniah", "Nahum", "Habakkuk", "Lamentations",
    "Ezekiel", "Obadiah", "Joel", "Jonah", "Malachi", "Haggai", "Ezra",
    "Nehemiah", "Esther", "Ecclesiastes"]}
RANGED = {
    "Isaiah_1":    [("Isaiah", [(1, 39)])],
    "Isaiah_2":    [("Isaiah", [(40, 55)])],
    "Isaiah_3":    [("Isaiah", [(56, 66)])],
    "Zechariah_1": [("Zechariah", [(1, 8)])],
    "Zechariah_2": [("Zechariah", [(9, 14)])],
    "Chronicles":  [("1_Chronicles", [(1, 29)]), ("2_Chronicles", [(1, 36)])],
    "Daniel":      [("Daniel", [(1, 1), (8, 12)])],
    "Jer_oracle":  [("Jeremiah", [(1,6),(8,10),(12,16),(19,20),(22,23),(30,31),(46,51)])],
}
# BHSA book names differ from the manifest labels for some books
BHSA_NAME = {"Numbers": "Numeri", "Deuteronomy": "Deuteronomium",
             "Ecclesiastes": "Ecclesiastes", "Lamentations": "Threni",
             "Ezekiel": "Ezechiel", "Obadiah": "Obadia", "Joel": "Joel",
             "Jonah": "Jona", "Malachi": "Maleachi", "Haggai": "Haggai",
             "Zephaniah": "Zephania", "Habakkuk": "Habakuk", "Nahum": "Nahum",
             "Micah": "Micha", "Hosea": "Hosea", "Amos": "Amos",
             "Isaiah": "Jesaia", "Jeremiah": "Jeremia", "Zechariah": "Sacharia",
             "Esther": "Esther", "Ezra": "Esra", "Nehemiah": "Nehemia",
             "Daniel": "Daniel", "1_Chronicles": "Chronica_I",
             "2_Chronicles": "Chronica_II", "Judges": "Judices",
             "Genesis": "Genesis", "Exodus": "Exodus", "Leviticus": "Leviticus"}


def unit_specs():
    man = json.load(open(DH.f("corpus_manifest_v2.json")))
    out = {}
    for k in ("training", "holdouts"):
        for t in man[k]:
            uid = t["id"]
            if uid in RANGED:
                spec_ = [(BHSA_NAME.get(b, b), r) for b, r in RANGED[uid]]
            elif uid in WHOLE:
                spec_ = [(BHSA_NAME.get(uid, uid), [(1, 999)])]
            else:
                print(f"  WARNING: no spec for {uid}"); continue
            out[uid] = dict(spec=spec_, date=t["date_bce"], sigma=t["date_sigma"],
                            genre=t.get("genre"), register=t.get("register"))
    return out


def main(target):
    api = ex.load_bhsa(TF_PATH)
    F, L, T = api.F, api.L, api.T
    units = unit_specs()
    rows = []
    for uid, meta in units.items():
        ex.words_for_ranges = ORIG_WFR      # undo any patch from the previous unit
        words = []
        for book, ranges in meta["spec"]:
            w = ORIG_WFR(book, ranges, F, L, T)
            if not w:
                print(f"  !! no words for {uid} / {book}")
            words.extend(w)
        if not words:
            continue
        # group words by verse so chunks never split a verse
        verses, cur, curv = [], [], None
        for w in words:
            v = L.u(w, "verse")
            v = v[0] if v else None
            if v != curv and cur:
                verses.append(cur); cur = []
            curv = v; cur.append(w)
        if cur:
            verses.append(cur)

        # accumulate verses into chunks of >= target words
        chunks, buf = [], []
        for v in verses:
            buf.extend(v)
            if len(buf) >= target:
                chunks.append(buf); buf = []
        if buf:
            if chunks and len(buf) < target * 0.5:
                chunks[-1].extend(buf)          # avoid a runt final chunk
            else:
                chunks.append(buf)

        for i, ch in enumerate(chunks):
            ex.words_for_ranges = lambda *a, _c=ch, **k: _c   # patch
            f = ex.extract_unit(f"{uid}_c{i}", [("x", [])], F, L, T)
            if f is None:
                continue
            f = {k: v for k, v in f.items() if k not in ("id", "label", "n_words")}
            rows.append(dict(chunk_id=f"{uid}_c{i:03d}", unit=uid,
                             date_bce=meta["date"], sigma=meta["sigma"],
                             genre=meta["genre"], register=meta["register"],
                             n_words=len(ch), **f))
        ex.words_for_ranges = ORIG_WFR
        print(f"  {uid:<15} {len(words):6d} words -> {len(chunks):3d} chunks")
    D = pd.DataFrame(rows)
    out = DH.f(f"chunk_features_{target}.csv")
    D.to_csv(out, index=False)
    print(f"\n{len(D)} chunks from {D.unit.nunique()} units -> {out}")
    return D


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500)
    a = ap.parse_args()
    main(a.target)
