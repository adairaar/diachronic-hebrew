#!/usr/bin/env python3
"""
build_feature_matrix_v2.py — rebuild the Hebrew feature matrix from BHSA
========================================================================
Rebuilds hebrew/data/feature_matrix_v2.csv for corpus v2.

Why this exists
---------------
An audit (2026-08) found that manuscript Table 1, hebrew/corpus_manifest.json,
and feature_rates_training.csv described three different corpora.  Five books
with defensible anchors (Obadiah, Joel, Zechariah 9-14, Lamentations,
Ecclesiastes) were absent from training; Jeremiah whole-book was in training
while its own oracle stratum served as the holdout, which is leakage.

This script rebuilds everything from BHSA in one pass so there is exactly one
authoritative feature matrix.

Reuses extract_unit() from hierarchical_bayes/00_extract_features.py so the
feature definitions are guaranteed identical to the existing pipeline.

Output: hebrew/data/feature_matrix_v2.csv
"""

from __future__ import annotations
import os, sys, json, importlib.util
import numpy as np
import pandas as pd

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.dirname(HERE)
HB     = os.path.join(HERE, "hierarchical_bayes")
TF_PATH = os.path.join(os.path.expanduser("~"),
                       "text-fabric-data", "github", "ETCBC", "bhsa", "tf", "2021")
MANIFEST = os.path.join(HERE, "corpus_manifest_v2.json")
OUT_CSV  = os.path.join(HERE, "data", "feature_matrix_v2.csv")

# ── Import the canonical extractor ────────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "extract00", os.path.join(HB, "00_extract_features.py"))
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

# ── Chapter ranges ────────────────────────────────────────────────────────────
# Whole books: empty range list means "all chapters".
WHOLE = {
    "Amos": "Amos", "Hosea": "Hosea", "Micah": "Micah",
    "Zephaniah": "Zephaniah", "Nahum": "Nahum", "Habakkuk": "Habakkuk",
    "Lamentations": "Lamentations", "Ezekiel": "Ezekiel",
    "Obadiah": "Obadiah", "Joel": "Joel", "Jonah": "Jonah",
    "Malachi": "Malachi", "Haggai": "Haggai",
    "Ezra": "Ezra", "Nehemiah": "Nehemiah",
    "Esther": "Esther", "Ecclesiastes": "Ecclesiastes",
    "Genesis": "Genesis", "Exodus": "Exodus",
    "Leviticus": "Leviticus", "Numbers": "Numbers",
    "Deuteronomy": "Deuteronomy",
}

# Chronicles = 1+2 Chronicles; Isaiah and Zechariah split by chapter.
RANGED = {
    "Isaiah_1":     [("Isaiah",      [(1, 39)])],
    "Isaiah_2":     [("Isaiah",      [(40, 55)])],
    "Isaiah_3":     [("Isaiah",      [(56, 66)])],
    "Zechariah_1":  [("Zechariah",   [(1, 8)])],
    "Zechariah_2":  [("Zechariah",   [(9, 14)])],
    "Chronicles":   [("1_Chronicles", [(1, 29)]), ("2_Chronicles", [(1, 36)])],
    "Daniel":       [("Daniel",      [(1, 1), (8, 12)])],          # Hebrew chapters only
    "Jer_oracle":   [("Jeremiah",    [(1,6),(8,10),(12,16),(19,20),(22,23),(30,31),(46,51)])],
    "Jer_DTR":      [("Jeremiah",    [(7,7),(11,11),(17,18),(21,21),(24,29),(32,45),(52,52)])],
    # Documentary sub-sources
    "D_Code":       [("Deuteronomy", [(12, 26)])],
    "D_Frame":      [("Deuteronomy", [(1, 11), (27, 31), (33, 34)])],
    "D_Song":       [("Deuteronomy", [(32, 32)])],
    "Lev_Holiness": [("Leviticus",   [(17, 26)])],
    "Lev_Priestly": [("Leviticus",   [(1, 16)])],
    "Song_Sea":     [("Exodus",      [(15, 15)])],
    "Song_Deborah": [("Judges",      [(5, 5)])],
}

# Documentary source composites (standard critical partition, Baden 2012)
P_CHAPTERS = [
    ("Genesis",   [(1,1),(5,5),(6,6),(9,9),(11,11),(17,17),(23,23),(25,25),(35,35),(36,36),(46,46),(48,48)]),
    ("Exodus",    [(6,7),(12,12),(16,16),(25,31),(35,40)]),
    ("Leviticus", [(1,27)]),
    ("Numbers",   [(1,10),(13,20),(25,36)]),
]
JE_CHAPTERS = [
    ("Genesis",   [(2,4),(6,8),(12,16),(18,22),(24,34),(37,45),(49,50)]),
    ("Exodus",    [(1,5),(8,11),(13,14),(17,24),(32,34)]),
    ("Numbers",   [(11,12),(21,24)]),
]
D_CHAPTERS = [("Deuteronomy", [(1, 34)])]

COMPOSITES = {
    "P_source":  P_CHAPTERS,
    "JE_source": JE_CHAPTERS,
    "D_source":  D_CHAPTERS,
    "Gen_JE":    [("Genesis", [(2,4),(6,8),(12,16),(18,22),(24,34),(37,45),(49,50)])],
    "Exo_JE":    [("Exodus",  [(1,5),(8,11),(13,14),(17,24),(32,34)])],
    "Num_JE":    [("Numbers", [(11,12),(21,24)])],
}


def all_chapter_span(book, F, L, T):
    """Return [(1, maxch)] for a whole book."""
    bn = T.nodeFromSection((book,))
    if bn is None:
        return None
    chs = [int(F.chapter.v(c)) for c in L.d(bn, "chapter")]
    return [(min(chs), max(chs))] if chs else None


def main():
    manifest = json.load(open(MANIFEST))
    api = ex.load_bhsa(TF_PATH)
    F, L, T = api.F, api.L, api.T

    # Metadata lookup from the manifest
    meta = {}
    for grp, hold in (("training", False), ("holdouts", True)):
        for t in manifest.get(grp, []):
            meta[t["id"]] = dict(
                date_bce=t["date_bce"], date_sigma=t["date_sigma"],
                register=t["register"], genre=t["genre"],
                holdout=hold, hbvi_holdout=bool(t.get("hbvi_holdout", False)))

    # Assemble the full unit list
    units: dict[str, list] = {}
    for uid, book in WHOLE.items():
        span = all_chapter_span(book, F, L, T)
        if span is None:
            print(f"  !! book not found in BHSA: {book}")
            continue
        units[uid] = [(book, span)]
    units.update(RANGED)
    units.update(COMPOSITES)

    print(f"\nExtracting {len(units)} units from BHSA …\n")
    rows = []
    for uid, pairs in units.items():
        feats = ex.extract_unit(uid, pairs, F, L, T)
        if feats is None:
            continue
        n_words = feats.pop("n_words")
        m = meta.get(uid, {})
        row = dict(feats)
        row.update(
            id=uid, n_words=n_words,
            date_bce=m.get("date_bce", np.nan),
            date_sigma=m.get("date_sigma", np.nan),
            register=m.get("register", ""),
            genre=m.get("genre", ""),
            holdout=m.get("holdout", False),
            hbvi_holdout=m.get("hbvi_holdout", False),
            in_training=uid in meta and not m.get("holdout", False),
        )
        rows.append(row)
        tag = ""
        if m.get("holdout"):      tag = "  [HOLDOUT both]"
        elif m.get("hbvi_holdout"): tag = "  [HB-VI holdout]"
        elif uid not in meta:     tag = "  [target/diagnostic]"
        print(f"  {uid:16s} {n_words:7d} w  "
              f"frac_ani={feats.get('frac_ani', float('nan')):.3f} "
              f"wayq={feats.get('rate_wayyiqtol', float('nan')):6.1f}/1k"
              f"{tag}")

    df = pd.DataFrame(rows)
    meta_first = ["id", "date_bce", "date_sigma", "register", "genre",
                  "holdout", "hbvi_holdout", "in_training", "n_words"]
    cols = meta_first + [c for c in df.columns if c not in meta_first]
    df = df[cols].set_index("id")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV)

    n_train = int(df["in_training"].sum())
    n_hold  = int(df["holdout"].sum())
    print(f"\nSaved → {OUT_CSV}")
    print(f"  training units      : {n_train}")
    print(f"  holdouts (both)     : {n_hold}")
    print(f"  HB-VI-only holdouts : {int(df['hbvi_holdout'].sum())}")
    print(f"  targets/diagnostic  : {len(df) - n_train - n_hold}")
    print(f"  feature columns     : {len([c for c in df.columns if c not in meta_first])}")


if __name__ == "__main__":
    main()
