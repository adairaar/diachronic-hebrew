"""
Generate numbers.tex: every quantitative claim in the manuscript as a LaTeX
macro read straight from a result file.

Nothing in main.tex is a hand-typed number.  If a pipeline is re-run and a
value changes, the manuscript changes with it; if a result file is missing,
this script fails loudly rather than leaving a stale figure in the text.
"""
import json, os, sys
import numpy as np, pandas as pd

R = "/home/claude"
G = "/home/claude/greek"
OUT = "/home/claude/ms/numbers.tex"
M = {}


def need(path):
    if not os.path.exists(path):
        sys.exit(f"MISSING RESULT FILE: {path}")
    return path


def put(name, value, fmt="{}"):
    """Define a LaTeX macro.  Names must be letters only."""
    assert name.isalpha(), name
    M[name] = fmt.format(value)


def signed(x, d=0):
    return f"{x:+.{d}f}"


# ══════════════════════════════════════════════════════════════════════
# 1. Hebrew corpus and model
# ══════════════════════════════════════════════════════════════════════
hb = json.load(open(need(f"{R}/final_lobo_metrics.json")))
put("HBfeat", hb["n_feats"])
put("HBchunks", hb["n_chunks"])
put("HBbooks", hb["n_books"])
put("HBwords", f"{hb['n_words']:,}")
put("HBscale", hb["S"], "{:.2f}")
put("HBmae", hb["mae"], "{:.0f}")
put("HBmaebase", hb["mae_baseline"], "{:.0f}")
put("HBrho", signed(hb["rho"], 2))
put("HBrhop", hb["rho_p"], "{:.4f}")
put("HBpair", hb["pair"] * 100, "{:.1f}")
put("HBnpair", hb["n_pair"])
put("HBqsixeight", hb["q68"], "{:.0f}")
put("HBqninety", hb["q90"], "{:.0f}")
put("HBnpre", hb["n_pre"]); put("HBnpreok", hb["n_pre_ok"])
put("HBnpost", hb["n_post"]); put("HBnpostok", hb["n_post_ok"])
put("HBsideok", hb["n_pre_ok"] + hb["n_post_ok"])
put("HBnperm", hb["n_perm"])
put("HBpermrho", hb["p_rho"], "{:.3f}")
put("HBpermpair", hb["p_pair"], "{:.3f}")
put("HBpermmae", hb["p_mae"], "{:.3f}")
put("HBnullrho", signed(hb["null_rho_med"], 2))
put("HBnullpair", hb["null_pair_med"] * 100, "{:.1f}")
put("HBnullmae", hb["null_mae_med"], "{:.0f}")

B = pd.read_csv(need(f"{R}/final_lobo_books.csv"))
put("HBoldest", int(B.truth.max())); put("HBlatest", int(B.truth.min()))
put("HBworstbook", B.loc[B.resid.abs().idxmax(), "book"].replace("_", " "))
put("HBworsterr", int(abs(B.resid).max()))
lam_ = B[B.book == "Lamentations"].iloc[0]
put("lamTruth", int(lam_.truth)); put("lamPred", int(lam_.pred))
put("lamErr", int(abs(lam_.resid)))

LK = pd.read_csv(need(f"{R}/leakage_generative.csv"))
tight = LK[LK.sigma_u <= 20]
put("leakShare", LK.data_share.median(), "{:.1f}")
put("leakLeaky", LK.err_leaky.mean(), "{:.0f}")
put("leakHonest", LK.err_honest.mean(), "{:.0f}")
put("leakTightN", len(tight))
put("leakTightShare", tight.data_share.mean(), "{:.1f}")
put("leakTightLeaky", tight.err_leaky.mean(), "{:.1f}")
put("leakTightHonest", tight.err_honest.mean(), "{:.0f}")

uc = json.load(open(need(f"{R}/uncal_compare.json")))
put("UNCALmae", uc["uncal"]["mae"], "{:.0f}")
put("UNCALspan", uc["uncal"]["span"], "{:.0f}")
put("UNCALpreok", uc["uncal"]["pre_ok"])
put("VARspan", uc["var"]["span"], "{:.0f}")
put("TRUEspan", uc["true_span"], "{:.0f}")

# ══════════════════════════════════════════════════════════════════════
# 2. Targets
# ══════════════════════════════════════════════════════════════════════
T = pd.read_csv(need(f"{R}/target_predictions_final.csv")).set_index("unit")
put("NTargets", len(T))
for u, mac in [("P_source", "P"), ("JE_source", "JE"), ("D_source", "D"),
               ("D_Code", "DCode"), ("D_Frame", "DFrame"),
               ("Lev_Holiness", "Hol"), ("Lev_Priestly", "LevP"),
               ("Jer_DTR", "JerD"), ("Gen_JE", "GenJE"), ("Exo_JE", "ExoJE"),
               ("Num_JE", "NumJE")]:
    r = T.loc[u]
    put(f"src{mac}", int(round(r.pred)))
    put(f"src{mac}lo", int(round(r.hi68)))     # hi68 = earlier BCE date
    put(f"src{mac}hi", int(round(r.lo68)))
    put(f"src{mac}post", r.p_post, "{:.2f}")
    put(f"src{mac}n", int(r.n_chunks))
put("srcminpost", T.loc[["P_source", "JE_source", "D_source"], "p_post"].min(),
    "{:.2f}")

# ══════════════════════════════════════════════════════════════════════
# 3. Poems (verse-precise)
# ══════════════════════════════════════════════════════════════════════
P = pd.read_csv(need(f"{R}/poem_predictions.csv")).set_index("unit")
for u, mac in [("SongSea_poem", "Sea"), ("SongSea_chapter", "SeaChap"),
               ("SongSea_prose", "SeaProse"), ("SongMoses_poem", "Moses"),
               ("SongMoses_prose", "MosesProse"), ("SongDeborah_poem", "Deb")]:
    r = P.loc[u]
    put(f"poem{mac}", int(round(r.pred)))
    put(f"poem{mac}lo", int(round(r.hi68)))
    put(f"poem{mac}hi", int(round(r.lo68)))
    put(f"poem{mac}post", r.p_post, "{:.2f}")
    put(f"poem{mac}w", int(r.n_words))
put("poemSeaGap", int(round(P.loc["SongSea_poem"].pred - P.loc["SongSea_prose"].pred)))
put("poemSeaProseFrac", 100 * P.loc["SongSea_prose"].n_words
    / P.loc["SongSea_chapter"].n_words, "{:.0f}")
put("poemMosesProseFrac", 100 * P.loc["SongMoses_prose"].n_words
    / P.loc["SongMoses_chapter"].n_words, "{:.0f}")

# ══════════════════════════════════════════════════════════════════════
# 4. Synthetic archaizing
# ══════════════════════════════════════════════════════════════════════
A = pd.read_csv(need(f"{R}/archaize_results.csv"))
units = list(A.unit.unique())
shifts, dens = {}, {}
for u in units:
    s = A[A.unit == u]
    a = float(s[s.rate == 0].pred.iloc[0]); b = float(s[s.rate == 1.0].pred.iloc[0])
    shifts[u] = b - a
    dens[u] = float(s[s.rate == 1.0].n_sub.iloc[0]) / float(s.n_words.iloc[0]) * 1000
put("arcN", len(units))
put("arcMean", np.mean(list(shifts.values())), "{:+.0f}")
put("arcMeanAbs", abs(np.mean(list(shifts.values()))), "{:.0f}")
put("arcMax", max(shifts.values()), "{:+.0f}")
put("arcMin", min(shifts.values()), "{:+.0f}")
put("arcTotalSwaps", int(A[A.rate == 1.0].n_sub.sum()))
# informative subset: books with real substitution density
inf = [u for u in units if dens[u] >= 5]
put("arcInfN", len(inf))
put("arcRate", np.mean([shifts[u] / dens[u] for u in inf]), "{:.1f}")
put("arcEccShift", shifts["Ecclesiastes"], "{:+.0f}")
put("arcEccDens", dens["Ecclesiastes"], "{:.1f}")
put("arcEzraTokens", int(A[(A.unit == "Ezra") & (A.rate == 1.0)].n_sub.iloc[0]))

# feature sensitivity, recomputed from the fitted model (sensitivity.py)
SS = json.load(open(need(f"{R}/sensitivity.json")))
put("sensLexShare", SS["lex_share"], "{:.0f}")
put("sensLexCount", SS["lex_count"])
put("sensTopLever", SS["top_lever"], "{:.1f}")
put("sensSD", SS["sea_gap_sd"], "{:.1f}")
put("sensDim", SS["n_feats"])
put("sensMorphShare", 100 - SS["lex_share"], "{:.0f}")

# ══════════════════════════════════════════════════════════════════════
# 5. Greek
# ══════════════════════════════════════════════════════════════════════
gk = json.load(open(need(f"{G}/greek_metrics.json")))
put("GKfeat", gk["n_feats"])
put("GKtexts", gk["n_train_texts"])
put("GKchunks", gk["n_train_chunks"])
put("GKatt", gk["n_att_texts"])
put("GKattchunks", gk["n_att_chunks"])
put("GKmae", gk["mae"], "{:.0f}")
put("GKmaebase", gk["mae_baseline"], "{:.0f}")
put("GKrho", signed(gk["rho"], 2))
put("GKrhop", gk["rho_p"], "{:.2g}")
put("GKpair", gk["pair"] * 100, "{:.1f}")
put("GKqsixeight", gk["q68"], "{:.0f}")
put("GKattshift", gk["att_mean_shift"], "{:+.0f}")
put("GKattshiftAbs", abs(gk["att_mean_shift"]), "{:.0f}")
put("GKattmed", gk["att_median_shift"], "{:+.0f}")
put("GKattearly", gk["att_n_early"])
GA = pd.read_csv(need(f"{G}/greek_atticizers.csv"))
put("GKattmaxearly", int(GA["shift"].min()))
put("GKattmaxname", GA.loc[GA["shift"].idxmin(), "author"])

big = GA[GA.n_chunks >= 40]
put("GKattshiftW", np.average(GA["shift"], weights=GA.n_chunks), "{:+.0f}")
put("GKattbign", len(big))
put("GKattshiftBig", big["shift"].mean(), "{:+.0f}")

RV = json.load(open(need(f"{G}/greek_register_variants.json")))
vb, vc = RV[1], RV[2]
put("GKattshiftB", vb["mean_shift"], "{:+.0f}")
put("GKattshiftC", vc["mean_shift"], "{:+.0f}")
put("GKmaeB", vb["mae"], "{:.0f}")
put("GKrhoB", signed(vb["rho"], 2))
put("GKattearlyTot", sum(v["n_early"] for v in RV))
put("GKattTot", sum(v["n_test"] for v in RV))

# ══════════════════════════════════════════════════════════════════════
with open(OUT, "w") as fh:
    fh.write("% AUTO-GENERATED by make_numbers.py -- do not edit\n")
    for k in sorted(M):
        fh.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
print(f"wrote {len(M)} macros -> {OUT}")
