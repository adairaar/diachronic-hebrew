"""
00_extract_features.py  —  BHSA feature extraction for HB-VI pipeline
=======================================================================
Extracts the full 69-feature vector (matching hebrew/data/feature_matrix.csv)
for the following units from the ETCBC/BHSA corpus:

  Jer_oracle   — Jeremiah without Deuteronomistic additions
                 chapters: 1–6, 8–10, 12–16, 19–20, 22–23, 30–31, 46–51
  Jer_DTR      — Deuteronomistic additions to Jeremiah (ch. 7,11,17–18,
                 21,24–29,32–45,52) — included for diagnostic comparison
  D_Code       — Deuteronomy ch. 12–26 (Deuteronomic Code)
  D_Frame      — Deuteronomy ch. 1–11, 27–31, 33–34 (framing speeches)
  D_Song       — Deuteronomy ch. 32 (Song of Moses)
  Lev_Holiness — Leviticus ch. 17–26 (Holiness Code)
  Lev_Priestly — Leviticus ch. 1–16 (core Priestly laws)
  Song_Sea     — Exodus ch. 15 (Song of the Sea)
  Song_Deborah — Judges ch. 5 (Song of Deborah)

Output: hebrew/hierarchical_bayes/results/extracted_features.csv
        (one row per unit, columns match feature_matrix.csv feature columns)

Usage:  python3 00_extract_features.py
Needs:  text-fabric + ETCBC/BHSA 2021 data
"""

from __future__ import annotations
import os, sys
import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pip install pandas --break-system-packages")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
PARENT  = os.path.dirname(HERE)
ROOT    = os.path.dirname(PARENT)
TF_PATH = os.path.join(os.path.expanduser("~"),
          "text-fabric-data", "github", "ETCBC", "bhsa", "tf", "2021")
FEAT_CSV = os.path.join(PARENT, "data", "feature_matrix.csv")
RESULTS  = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# ── Chapter-range definitions ─────────────────────────────────────────────────
UNITS: dict[str, list[tuple[str, list[tuple[int,int]]]]] = {
    "Jer_oracle":    [("Jeremiah",    [(1,6),(8,10),(12,16),(19,20),(22,23),(30,31),(46,51)])],
    "Jer_DTR":       [("Jeremiah",    [(7,7),(11,11),(17,18),(21,21),(24,29),(32,45),(52,52)])],
    "D_Code":        [("Deuteronomy", [(12,26)])],
    "D_Frame":       [("Deuteronomy", [(1,11),(27,31),(33,34)])],
    "D_Song":        [("Deuteronomy", [(32,32)])],
    "Lev_Holiness":  [("Leviticus",   [(17,26)])],
    "Lev_Priestly":  [("Leviticus",   [(1,16)])],
    "Song_Sea":      [("Exodus",      [(15,15)])],
    "Song_Deborah":  [("Judges",      [(5,5)])],
}

# ── Clause-type sets (from script 14 / ETCBC conventions) ────────────────────
WAYQ_TYPES   = {"Way0","WayX","WaYX"}
WQTL_TYPES   = {"WQt0","WQtX","WxQ0","WxQX"}
WNARR_TYPES  = WAYQ_TYPES | WQTL_TYPES
FRONT_TYPES  = {"xQt0","xYq0","xQtX","xYqX","xIm0",
                "WxY0","WxQ0","WxQX","WxYX","WxI0","XQtl","XYqt"}
CPEN_TYPES   = {"CPen"}
NMCL_TYPES   = {"NmCl","AjCl"}
PTCP_TYPES   = {"Ptcp"}
INFC_TYPES   = {"InfC"}
SKIP_TYPES   = {"Ellp","Voct","MSyn","InfA"}
VERBAL_TYPES = (WAYQ_TYPES | WQTL_TYPES | FRONT_TYPES |
                {"ZQt0","ZQtX","ZYq0","ZYqX","ZIm0",
                 "WYq0","WYqX","WIm0","WXYq","WXQt",
                 "xQt0","xYq0","xQtX","xYqX","Way0","WayX"})


# ── BHSA loader ───────────────────────────────────────────────────────────────
def load_bhsa(path: str):
    try:
        from tf.fabric import Fabric
    except ImportError:
        sys.exit("pip install text-fabric --break-system-packages")
    print(f"Loading BHSA from {path} …")
    TF = Fabric(locations=path, silent=True)
    api = TF.load(
        "otype lex sp vs vt nu gn st prs_ps chapter typ function",
        silent=True)
    return api


# ── Helpers ───────────────────────────────────────────────────────────────────
def words_for_ranges(book: str, ch_ranges, F, L, T):
    bn = T.nodeFromSection((book,))
    if bn is None:
        return []
    words = []
    for ch_node in L.d(bn, "chapter"):
        ch = int(F.chapter.v(ch_node))
        if any(s <= ch <= e for s, e in ch_ranges):
            words.extend(L.d(ch_node, "word"))
    return words


def extract_unit(label: str, book_ch_pairs, F, L, T) -> dict | None:
    """Return a feature dict for the given chapter ranges, or None if empty."""
    from collections import Counter

    all_words = []
    for book, ch_ranges in book_ch_pairs:
        all_words.extend(words_for_ranges(book, ch_ranges, F, L, T))

    if not all_words:
        print(f"  WARNING: no words found for {label}")
        return None

    n   = len(all_words)
    lc  = Counter(F.lex.v(w) for w in all_words)
    vb  = [w for w in all_words if F.sp.v(w) == "verb"]
    nb  = [w for w in all_words if F.sp.v(w) == "subs"]
    vtc = Counter(F.vt.v(w) for w in vb)
    vsc = Counter(F.vs.v(w) for w in vb)
    n_vb = len(vb)

    def lr(lex):   return lc.get(lex, 0) / n * 1000   # lex rate per 1k
    def vtr(vt):   return vtc.get(vt, 0) / n * 1000   # verb-tense rate per 1k
    def vsr(vs):   return vsc.get(vs, 0) / n * 1000   # verb-stem rate per 1k

    ani,  ank  = lc.get(">NJ",0),   lc.get(">NKJ",0)
    she,  ash  = lc.get("C",0),     lc.get(">CR",0)
    ain_, lo   = lc.get(">JN/",0),  lc.get("L>",0)
    al         = lc.get(">L",0)
    bqsh, shal = lc.get("BQC[",0),  lc.get("C>L[",0)
    qhl,  edh  = lc.get("QHL/",0),  lc.get("<DH/",0)
    anx,  nxnw = lc.get(">NXNW",0), lc.get("NXNW",0)

    # ── 44 theoretical features ───────────────────────────────────────────────
    feats: dict = {
        "frac_ani":     ani  / (ani+ank)   if ani+ank  > 0 else np.nan,
        "rate_anochi":  lr(">NKJ"),
        "rate_ani":     lr(">NJ"),
        "frac_she":     she  / (she+ash)   if she+ash  > 0 else np.nan,
        "rate_asher":   lr(">CR"),
        "frac_ein":     ain_ / (ain_+lo)   if ain_+lo  > 0 else np.nan,
        "rate_neg_al":  lr(">L"),
        "frac_neg_al":  al   / (al+lo)     if al+lo    > 0 else np.nan,
        "rate_neg_lo":  lr("L>"),
        "rate_ki":      lr("KJ"),
        "rate_gam":     lr("GM"),
        "rate_lakhen":  lr("LKN"),
        "rate_hinne":   lr("HNH"),
        "rate_atta":    lr("<TH"),
        "rate_az":      lr(">Z"),
        "rate_af":      lr(">P/"),
        "rate_wayyiqtol": vtr("wayq"),
        "rate_qatal":   vtr("perf"),
        "rate_yiqtol":  vtr("impf"),
        "rate_ptca":    vtr("ptca"),
        "rate_ptcp":    vtr("ptcp"),
        "rate_inf_abs": vtr("infa"),
        "rate_inf_con": vtr("infc"),
        "rate_impv":    vtr("impv"),
        "rate_jussive": vtr("juss"),
        "rate_cohort":  vtr("coht"),
        "rate_qal":     vsr("qal"),
        "rate_hiphil":  vsr("hif"),
        "rate_piel":    vsr("piel"),
        "rate_niphal":  vsr("nif"),
        "rate_hithpael":vsr("hit"),
        "rate_hophal":  vsr("hof"),
        "frac_baqash":  bqsh / (bqsh+shal) if bqsh+shal > 0 else np.nan,
        "rate_baqash":  lr("BQC["),
        "rate_shaal":   lr("C>L["),
        "frac_qahal":   qhl  / (qhl+edh)   if qhl+edh  > 0 else np.nan,
        "rate_hayah":   lr("HJH["),
        "rate_amar":    lr(">MR["),
        "rate_natan":   lr("NTN["),
        "rate_halak":   lr("HLK["),
        "rate_const":   sum(1 for w in nb if F.st.v(w) == "c") / n * 1000,
        "rate_prs":     sum(1 for w in all_words if F.prs_ps.v(w) != "NA") / n * 1000,
        "rate_pl_noun": sum(1 for w in nb if F.nu.v(w) == "pl") / n * 1000,
        "rate_f_noun":  sum(1 for w in nb if F.gn.v(w) == "f")  / n * 1000,
        # ── 12 morpho features ────────────────────────────────────────────────
        "frac_anachnu": anx / (anx+nxnw)  if anx+nxnw > 0 else np.nan,
        "rate_pen":     lr("PN"),
        "rate_terem":   lr("TRM"),
        "rate_na":      lr("N>"),
        "frac_halak_piel": (sum(1 for w in vb if F.lex.v(w)=="HLK[" and F.vs.v(w)=="piel")
                            / lc.get("HLK[",0) if lc.get("HLK[",0) > 0 else np.nan),
        "frac_ysf_qal": (sum(1 for w in vb if F.lex.v(w)=="JSP[" and F.vs.v(w)=="qal")
                         / lc.get("JSP[",0)  if lc.get("JSP[",0)  > 0 else np.nan),
        "frac_niphal":  vsc.get("nif",0) / n_vb if n_vb > 0 else np.nan,
        "frac_yld_nif": (sum(1 for w in vb if F.lex.v(w)=="JLD[" and F.vs.v(w)=="nif")
                         / lc.get("JLD[",0) if lc.get("JLD[",0) > 0 else np.nan),
        "frac_zaqaq":   np.nan,   # ZQQ[ is too rare for reliable fraction computation
        "rate_ut_nouns": sum(v for k,v in lc.items()
                             if k.endswith("WT/") or k.endswith("WT")) / n * 1000,
    }

    # ── 10 tier-3 clause features ─────────────────────────────────────────────
    n_cls = n_nmcl = n_front = n_cpen = n_ptcp_cl = n_infc_cl = 0
    n_wayq_cl = n_wqtl_cl = n_wnarr_cl = n_verbal = 0
    n_sv = n_sv_tot = n_ov = n_ov_tot = n_null = 0

    for book, ch_ranges in book_ch_pairs:
        bn = T.nodeFromSection((book,))
        if bn is None:
            continue
        for ch_node in L.d(bn, "chapter"):
            ch = int(F.chapter.v(ch_node))
            if not any(s <= ch <= e for s, e in ch_ranges):
                continue
            for cl in L.d(ch_node, "clause"):
                typ = F.typ.v(cl)
                if typ in SKIP_TYPES:
                    continue
                n_cls += 1
                if typ in NMCL_TYPES:   n_nmcl    += 1
                if typ in FRONT_TYPES:  n_front   += 1
                if typ in CPEN_TYPES:   n_cpen    += 1
                if typ in PTCP_TYPES:   n_ptcp_cl += 1
                if typ in INFC_TYPES:   n_infc_cl += 1
                if typ in WAYQ_TYPES:   n_wayq_cl += 1
                if typ in WQTL_TYPES:   n_wqtl_cl += 1
                if typ in WNARR_TYPES:  n_wnarr_cl += 1
                if typ in VERBAL_TYPES:
                    n_verbal += 1
                    phs = list(L.d(cl, "phrase"))
                    pf  = {F.function.v(ph): ph for ph in phs}
                    if "Subj" not in pf:
                        n_null += 1
                    if "Subj" in pf and "Pred" in pf:
                        n_sv_tot += 1
                        if pf["Subj"] < pf["Pred"]:
                            n_sv += 1
                    if "Objc" in pf and "Pred" in pf:
                        n_ov_tot += 1
                        if pf["Objc"] < pf["Pred"]:
                            n_ov += 1

    feats.update({
        "frac_nmcl":      n_nmcl    / n_cls      if n_cls      > 0  else np.nan,
        "frac_fronted":   n_front   / n_cls      if n_cls      > 0  else np.nan,
        "frac_cpen":      n_cpen    / n_cls      if n_cls      > 0  else np.nan,
        "frac_ptcp_cl":   n_ptcp_cl / n_cls      if n_cls      > 0  else np.nan,
        "frac_infc":      n_infc_cl / n_cls      if n_cls      > 0  else np.nan,
        "frac_wqtl_wayq": n_wqtl_cl/ n_wnarr_cl if n_wnarr_cl > 5  else np.nan,
        "rate_cpen_1k":   n_cpen    / n * 1000   if n          > 0  else np.nan,
        "frac_sv":        n_sv      / n_sv_tot   if n_sv_tot   > 0  else np.nan,
        "frac_null_subj": n_null    / n_verbal   if n_verbal   > 10 else np.nan,
        "frac_ov":        n_ov      / n_ov_tot   if n_ov_tot   > 0  else np.nan,
    })

    feats["n_words"] = n
    return feats


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    api = load_bhsa(TF_PATH)
    F = api.F; L = api.L; T = api.T

    # Get feature column order from existing feature matrix
    fm = pd.read_csv(FEAT_CSV, index_col="id")
    meta_cols = {"date_bce", "date_sigma", "register", "genre", "holdout"}
    feat_cols  = [c for c in fm.columns if c not in meta_cols]

    print(f"\nExtracting features for {len(UNITS)} units …")
    rows = []
    for label, pairs in UNITS.items():
        feats = extract_unit(label, pairs, F, L, T)
        if feats is None:
            continue
        n_words = feats.pop("n_words")
        row = {col: feats.get(col, np.nan) for col in feat_cols}
        row["id"]      = label
        row["n_words"] = n_words
        rows.append(row)
        print(f"  {label:20s}  {n_words:6d} words  "
              f"frac_ani={feats.get('frac_ani',float('nan')):.3f}  "
              f"wayyiqtol={feats.get('rate_wayyiqtol',float('nan')):.1f}/1k  "
              f"frac_she={feats.get('frac_she',float('nan')):.3f}")

    out = pd.DataFrame(rows).set_index("id")
    out_path = os.path.join(RESULTS, "extracted_features.csv")
    out.to_csv(out_path)
    print(f"\nSaved → {out_path}")

    # Diagnostic: cross-check Jer_oracle vs Jer_DTR on key features
    if "Jer_oracle" in out.index and "Jer_DTR" in out.index:
        print("\n── Jer_oracle vs Jer_DTR diagnostic ─────────────────────")
        for feat in ["frac_ani", "frac_she", "rate_wayyiqtol", "rate_neg_lo",
                     "rate_ut_nouns", "frac_niphal", "frac_infc"]:
            orc = out.loc["Jer_oracle", feat] if feat in out.columns else float("nan")
            dtr = out.loc["Jer_DTR",    feat] if feat in out.columns else float("nan")
            print(f"  {feat:20s}  oracle={orc:7.3f}  DTR={dtr:7.3f}")


if __name__ == "__main__":
    main()
