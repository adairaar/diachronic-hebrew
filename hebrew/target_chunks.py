"""
Chunk-level features for the UNDATED targets, using the same 634-feature
extractor as the dated corpus.
"""
import os, json, argparse, collections
import numpy as np, pandas as pd
import importlib.util

spec = importlib.util.spec_from_file_location("big", "/home/claude/chunk_extract_big.py")
big = importlib.util.module_from_spec(spec); spec.loader.exec_module(big)

B = "Chronica"  # placeholder to keep linters quiet
TARGETS = {
    "P_source":  [("Genesis",[(1,1),(5,5),(6,6),(9,9),(11,11),(17,17),(23,23),(25,25),(35,35),(36,36),(46,46),(48,48)]),
                  ("Exodus",[(6,7),(12,12),(16,16),(25,31),(35,40)]),
                  ("Leviticus",[(1,27)]),
                  ("Numeri",[(1,10),(13,20),(25,36)])],
    "JE_source": [("Genesis",[(2,4),(6,8),(12,16),(18,22),(24,34),(37,45),(49,50)]),
                  ("Exodus",[(1,5),(8,11),(13,14),(17,24),(32,34)]),
                  ("Numeri",[(11,12),(21,24)])],
    "D_source":  [("Deuteronomium",[(1,34)])],
    "Gen_JE":    [("Genesis",[(2,4),(6,8),(12,16),(18,22),(24,34),(37,45),(49,50)])],
    "Exo_JE":    [("Exodus",[(1,5),(8,11),(13,14),(17,24),(32,34)])],
    "Num_JE":    [("Numeri",[(11,12),(21,24)])],
    "D_Code":    [("Deuteronomium",[(12,26)])],
    "D_Frame":   [("Deuteronomium",[(1,11),(27,31),(33,34)])],
    "D_Song":    [("Deuteronomium",[(32,32)])],
    "Lev_Holiness":[("Leviticus",[(17,26)])],
    "Lev_Priestly":[("Leviticus",[(1,16)])],
    "Song_Sea":  [("Exodus",[(15,15)])],
    "Song_Deborah":[("Judices",[(5,5)])],
    "Jer_DTR":   [("Jeremia",[(7,7),(11,11),(17,18),(21,21),(24,29),(32,45),(52,52)])],
    "Genesis":   [("Genesis",[(1,50)])],
    "Exodus":    [("Exodus",[(1,40)])],
    "Leviticus": [("Leviticus",[(1,27)])],
    "Numbers":   [("Numeri",[(1,36)])],
    "Deuteronomy":[("Deuteronomium",[(1,34)])],
}


def main(target):
    api = big.load(); F, L = api.F, api.L
    allw = [w for w in F.otype.s("word") if F.language.v(w) == "Hebrew"]
    top_lex = [k for k, _ in collections.Counter(F.lex.v(w) for w in allw).most_common(250)]
    top_pos = [k for k, _ in collections.Counter(F.sp.v(w) for w in allw).most_common(14)]
    ph = list(F.otype.s("phrase")); cl = list(F.otype.s("clause"))
    top_typ = [k for k, _ in collections.Counter(F.typ.v(p) for p in ph).most_common(12)]
    top_fun = [k for k, _ in collections.Counter(F.function.v(p) for p in ph).most_common(16)]
    top_rela = [k for k, _ in collections.Counter(F.rela.v(c) for c in cl).most_common(12)]
    top_ctyp = [k for k, _ in collections.Counter(F.typ.v(c) for c in cl).most_common(14)]

    rows = []
    for uid, spec_ in TARGETS.items():
        words = big.unit_words(api, spec_)
        if not words:
            print(f"  !! no words for {uid}"); continue
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
            fe = big.chunk_feats(ch, api, top_lex, top_pos, top_typ, top_fun, top_rela, top_ctyp)
            rows.append(dict(chunk_id=f"{uid}_c{i:03d}", unit=uid, date_bce=np.nan,
                             genre=None, register=None, n_words=len(ch), **fe))
        print(f"  {uid:<15} {len(words):6d} w -> {len(chunks):3d} chunks")
    D = pd.DataFrame(rows)
    out = f"/home/claude/target_chunks_{target}.csv"
    D.to_csv(out, index=False)
    print(f"\n{len(D)} chunks from {D.unit.nunique()} targets -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--target", type=int, default=500)
    main(ap.parse_args().target)
