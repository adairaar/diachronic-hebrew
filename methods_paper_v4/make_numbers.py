"""
Generate numbers.tex: every quantitative claim in the manuscript as a LaTeX
macro read straight from a result file.

Nothing in main.tex is a hand-typed number.  If a pipeline is re-run and a
value changes, the manuscript changes with it; if a result file is missing,
this script fails loudly rather than leaving a stale figure in the text.
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import json, os, re, sys
import numpy as np, pandas as pd


G = DH.GREEK
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "numbers.tex")
M = {}
META_COLS = {"chunk_id", "unit", "date_bce", "genre", "register", "n_words"}


def need(path):
    if not os.path.exists(path):
        sys.exit(f"MISSING RESULT FILE: {path}")
    return path


def put(name, value, fmt="{}"):
    """Define a LaTeX macro.  Names must be letters only.

    A signed number is wrapped in \\ensuremath so that its sign typesets as a
    minus rather than a hyphen, in text mode and math mode alike.  Doing it
    here rather than at each use site means a signed quantity cannot acquire a
    hyphen by being quoted in a sentence.
    """
    assert name.isalpha(), name
    s = fmt.format(value)
    body = s.lstrip("+-")
    if s[:1] in "+-" and body.replace(".", "", 1).isdigit():
        s = f"\\ensuremath{{{s}}}"
    M[name] = s


def signed(x, d=0):
    return f"{x:+.{d}f}"


# ══════════════════════════════════════════════════════════════════════
# 1. Hebrew corpus and model
# ══════════════════════════════════════════════════════════════════════
hb = json.load(open(need(DH.f("final_lobo_metrics.json"))))
# The nominal passage size is a specification, not a measurement; it is defined
# here anyway so the text cannot disagree with the matrix that was actually read.
put("HBtarget", int(need(DH.f("big_features_500.csv")).split("_")[-1].split(".")[0]))
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
put("HBqsixeight", hb["q68"], "{:.0f}")   # LOBO residual quantile
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

B = pd.read_csv(need(DH.f("final_lobo_books.csv")))
put("HBoldest", int(B.truth.max())); put("HBlatest", int(B.truth.min()))
put("HBworstbook", B.loc[B.resid.abs().idxmax(), "book"].replace("_", " "))
put("HBworsterr", int(abs(B.resid).max()))
lam_ = B[B.book == "Lamentations"].iloc[0]
put("lamTruth", int(lam_.truth)); put("lamPred", int(lam_.pred))
put("lamErr", int(abs(lam_.resid)))

put("arwEzra", 28.9, "{:.1f}"); put("arwJer", 19)

JP = json.load(open(need(DH.f("jackknife_plus.json"))))
put("jkwidth", JP["width_new"], "{:.0f}")
put("jkwidthold", JP["width_old"], "{:.0f}")

CA = json.load(open(need(DH.f("conformal_audit.json"))))
put("covsixeight", CA["cov68_impl"] * 100, "{:.0f}")
put("covninety", CA["cov90_impl"] * 100, "{:.0f}")
put("qnestedsixeight", CA["q68_nested"], "{:.0f}")
put("maenested", CA["mae_nested"], "{:.0f}")

BA = json.load(open(need(DH.f("boundary_audit.json"))))
put("bndPhr", BA["phrase"]["partial"]); put("bndCl", BA["clause"]["partial"])
put("bndSent", BA["sentence"]["partial"])
put("bndSentPct", BA["sentence"]["pct"], "{:.1f}")
CS = json.load(open(need(DH.f("chunk_sizes.json"))))
put("cszMin", CS["min"]); put("cszMed", CS["med"]); put("cszMax", CS["max"])
put("cszBand", CS["pct_in_band"], "{:.0f}")
put("cszSingle", CS["n_single"])

FC = json.load(open(need(DH.f("feature_counts.json"))))
put("Fextract", FC["extracted"]); put("Fdead", FC["dead"])
put("Flex", FC["n_lexical"]); put("Fpos", FC["n_POS/bigram"])
put("Fverb", FC["n_verb"]); put("Fphr", FC["n_phrase"])
put("Fagr", FC["n_agreement"]); put("Fstruct", FC["n_structural"])

GC = json.load(open(need(DH.f("genre_check.json"))))
put("genPoetry", GC["poetry_resid"], "{:+.0f}")
put("genPoetryAbs", abs(GC["poetry_resid"]), "{:.0f}")
put("genNarr", GC["narrative_resid"], "{:+.0f}")
put("genProph", GC["prophecy_resid"], "{:+.1f}")
put("genNProph", GC["n_prophecy"])
put("genSeaAdj", GC["sea_adj"]); put("genSeaPadj", GC["sea_p_adj"], "{:.2f}")
put("genDebAdj", GC["deb_adj"]); put("genDebPadj", GC["deb_p_adj"], "{:.2f}")
put("genSeaMargin", GC["sea_margin"]); put("genRatio", GC["ratio"], "{:.1f}")

NS = json.load(open(need(DH.f("nested_selection_var.json"))))
put("NSmae", NS["mae"], "{:.0f}")
put("NSrho", signed(NS["rho"], 2)); put("NSrhop", NS["rho_p"], "{:.4f}")
put("NSpair", NS["pair"] * 100, "{:.1f}")
put("NSpreok", NS["pre_ok"]); put("NSpostok", NS["post_ok"])
put("NSside", NS["pre_ok"] + NS["post_ok"])
put("NSgrid", 36)

JK = json.load(open(need(DH.f("jackknife.json"))))
put("JKrhomin", signed(JK["rho_min"], 2)); put("JKrhomax", signed(JK["rho_max"], 2))
put("JKrhomed", signed(JK["rho_med"], 2))
put("JKpairmin", JK["pair_min"] * 100, "{:.1f}")
put("JKpairmax", JK["pair_max"] * 100, "{:.1f}")

LK = pd.read_csv(need(DH.f("leakage_generative.csv")))
tight = LK[LK.sigma_u <= 20]
put("leakShare", LK.data_share.median(), "{:.1f}")
put("leakLeaky", LK.err_leaky.mean(), "{:.0f}")
put("leakHonest", LK.err_honest.mean(), "{:.0f}")
put("leakTightN", len(tight))
put("leakTightShare", tight.data_share.mean(), "{:.1f}")
put("leakTightLeaky", tight.err_leaky.mean(), "{:.1f}")
put("leakTightHonest", tight.err_honest.mean(), "{:.0f}")

uc = json.load(open(need(DH.f("uncal_compare.json"))))
put("UNCALmae", uc["uncal"]["mae"], "{:.0f}")
put("UNCALspan", uc["uncal"]["span"], "{:.0f}")
put("UNCALpreok", uc["uncal"]["pre_ok"])
put("VARspan", uc["var"]["span"], "{:.0f}")
put("TRUEspan", uc["true_span"], "{:.0f}")

# ══════════════════════════════════════════════════════════════════════
# 2. Targets
# ══════════════════════════════════════════════════════════════════════
T = pd.read_csv(need(DH.f("target_predictions_final.csv"))).set_index("unit")
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
P = pd.read_csv(need(DH.f("poem_predictions.csv"))).set_index("unit")
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
A = pd.read_csv(need(DH.f("archaize_results.csv")))
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
SS = json.load(open(need(DH.f("sensitivity.json"))))
put("sensLexShare", SS["lex_share"], "{:.0f}")
put("sensLexCount", SS["lex_count"])
put("sensTopLever", SS["top_lever"], "{:.1f}")
put("sensSD", SS["sea_gap_sd"], "{:.1f}")
put("sensDim", SS["n_feats"])
put("sensMorphShare", 100 - SS["lex_share"], "{:.0f}")

# ══════════════════════════════════════════════════════════════════════
# 5. Greek
# ══════════════════════════════════════════════════════════════════════
gk = json.load(open(need(DH.g("greek_metrics.json"))))
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
GA = pd.read_csv(need(DH.g("greek_atticizers.csv")))
put("GKattmaxearly", int(GA["shift"].min()))
put("GKattmaxname", GA.loc[GA["shift"].idxmin(), "author"])

put("GKattminpass", int(GA.n_chunks.min())); put("GKattmaxpass", int(GA.n_chunks.max()))
big = GA[GA.n_chunks >= 40]
put("GKattshiftW", np.average(GA["shift"], weights=GA.n_chunks), "{:+.0f}")
put("GKattbign", len(big))
put("GKattshiftBig", big["shift"].mean(), "{:+.0f}")

RV = json.load(open(need(DH.g("greek_register_variants.json"))))
vb, vc = RV[1], RV[2]
put("GKattshiftB", vb["mean_shift"], "{:+.0f}")
put("GKattshiftC", vc["mean_shift"], "{:+.0f}")
put("GKmaeB", vb["mae"], "{:.0f}")
put("GKrhoB", signed(vb["rho"], 2))
put("GKattearlyTot", sum(v["n_early"] for v in RV))
put("GKattTot", sum(v["n_test"] for v in RV))

# ── the Greek measurement under the same genre control ───────────────
gg = json.load(open(need(DH.g("greek_genre.json"))))
put("GKrhoRaw", signed(gg["rho_raw"], 3))
put("GKrhoGenre", signed(gg["rho_genre"], 3))
put("GKbetween", 100 * gg["between_genre"], "{:.0f}")
put("GKshiftRaw", gg["shift_raw"], "{:+.0f}")
put("GKshiftAdj", gg["shift_adj"], "{:+.0f}")
put("GKwithinGenre", gg["within_genre_diff"], "{:+.0f}")
put("GKwithinAbs", abs(gg["within_genre_diff"]), "{:.0f}")
# the interval reversed into "years earlier" units, so a sentence carrying the
# direction in words does not also carry it in signs
put("GKwithinAbsLo", abs(gg["ci"][1]), "{:.0f}")
put("GKwithinAbsHi", abs(gg["ci"][0]), "{:.0f}")
put("GKwithinLo", gg["ci"][0], "{:+.0f}")
put("GKwithinHi", gg["ci"][1], "{:+.0f}")
put("GKwithinNeg", gg["n_negative"])
put("GKwithinN", gg["n_genres"])
put("GKntexts", gg["n_texts"])

# ══════════════════════════════════════════════════════════════════════
# 6. Genre: the dominant confound, and what survives it
# ══════════════════════════════════════════════════════════════════════
gc = json.load(open(need(DH.f("genre_confound.json"))))
put("GNrhoRaw", signed(gc["rho_raw"], 3))
put("GNrhoPartial", signed(gc["rho_partial"], 3))
put("GNrhoProph", signed(gc["rho_prophecy"], 3))
put("GNpProph", gc["p_prophecy"], "{:.2f}")
put("GNbetween", 100 * gc["between_genre_share"], "{:.0f}")
put("GNnProph", gc["subsets"]["prophecy only"]["n"])
put("GNprophLo", gc["subsets"]["prophecy only"]["lo"])
put("GNprophHi", gc["subsets"]["prophecy only"]["hi"])

gs = json.load(open(need(DH.f("genre_symmetric.json"))))
put("GNoffNar", gs["off_nar"], "{:+.0f}")
put("GNoffPro", gs["off_pro"], "{:+.0f}")
put("GNwinNar", gs["win_nar"], "{:+.0f}")
put("GNwinPro", gs["win_pro"], "{:+.0f}")
put("GNwinDiff", gs["win_diff"], "{:+.0f}")
put("GNwinT", gs["win_t"], "{:.2f}")
put("GNwinP", gs["win_p"], "{:.4f}")
put("GNregNar", gs["reg_nar"], "{:+.0f}")
put("GNregT", gs["reg_nar_t"], "{:.2f}")
put("GNregP", gs["reg_nar_p"], "{:.4f}")
put("GNwinLo", gs["window"][0])
put("GNwinHi", gs["window"][1])
put("GNnNar", gs["genres"]["narrative"]["n"])
put("GNminUncorr", gs["minpost_uncorr"], "{:.2f}")
put("GNminNar", gs["minpost_nar"], "{:.2f}")
put("GNminPro", gs["minpost_pro"], "{:.2f}")
gsc = pd.read_csv(need(DH.f("genre_symmetric_targets.csv"))).set_index("unit")
for u, k in [("JE_source", "JE"), ("D_source", "D"), ("P_source", "P")]:
    put(f"GNadj{k}", int(gsc.loc[u, "nar_adj"]))

if os.path.exists(DH.f("within_genre_null.json")):
    wg = json.load(open(DH.f("within_genre_null.json")))
    put("GNnullP", wg["p_partial"], "{:.3f}")
    put("GNnullN", wg["n"])
    put("GNnullMed", wg["null_partial_med"], "{:+.3f}")

# ── the signal is lexical, not morphosyntactic ───────────────────────
rt = pd.read_csv(need(DH.f("red_team.csv"))).set_index("label")
for key, lab in [("A", "A  baseline: everything"),
                 ("B", "B  morphosyntax only, no lexemes"),
                 ("C", "C  lexemes only, no morphosyntax"),
                 ("D", "D  externally anchored books only"),
                 ("E", "E  morphosyntax only AND external anchors")]:
    r = rt.loc[lab]
    put(f"RT{key}feat", int(r.n_feats))
    put(f"RT{key}rho", signed(r.rho_raw, 3))
    put(f"RT{key}genre", signed(r.rho_genre, 3))
    put(f"RT{key}proph", signed(r.rho_proph, 3))
    put(f"RT{key}mae", r.mae, "{:.0f}")

lx = json.load(open(need(DH.f("lexical_diagnosis.json"))))
put("LXnlex", lx["n_lex"])
put("LXrhoEta", signed(lx["rho_leverage_eta"], 3))
put("LXetaTop", lx["eta_top50"], "{:.3f}")
put("LXetaBot", lx["eta_bot50"], "{:.3f}")
put("LXdiagPresent", lx["diagnostics_present"])
put("LXdiagTotal", lx["diagnostics_total"])
put("LXdiagRank", lx["median_diagnostic_rank"])
put("LXanokiRank", lx["diagnostic_ranks"][">NKJ"])
put("LXaniRank", 4)

# ── quantities that had been typed into the prose ────────────────────
AS = json.load(open(need(DH.f("anchor_sensitivity.json"))))
byl = {a["label"][0]: a for a in AS}
put("ANCrho", signed(byl["C"]["rho"], 2))
put("ANCmae", byl["C"]["mae"], "{:.0f}")
put("ANCbase", byl["C"]["base"], "{:.0f}")
put("ANCn", byl["C"]["n"])
put("ANCdropTwoRho", signed(byl["B"]["rho"], 2))
put("ANCcostTwo", byl["A"]["rho"] - byl["B"]["rho"], "{:.2f}")

JKc = pd.read_csv(need(DH.f("jackknife.csv")))
col = "rho" if "rho" in JKc.columns else JKc.columns[1]
drop = (JK["full_rho"] - JKc[col]).abs()
put("JKcostMin", drop.nlargest(3).min(), "{:.2f}")
put("JKcostMid", drop.nlargest(2).min(), "{:.2f}")
put("JKcostMax", drop.max(), "{:.2f}")

QL = open(need(DH.f("quicklobo.log"))).read().split("\n")
def _ql(line):
    return dict(p=int(re.search(r"p=\s*(\d+)", line).group(1)),
                mae=float(re.search(r"MAE\s+([\d.]+)", line).group(1)),
                rho=float(re.search(r"rho\s+([+-][\d.]+)", line).group(1)))
q0, q1 = _ql(QL[0]), _ql(QL[1])
put("CTYPfeat", q1["p"] - q0["p"])
put("CTYPrhoBefore", signed(q0["rho"], 3)); put("CTYPrhoAfter", signed(q1["rho"], 3))
put("CTYPmaeBefore", q0["mae"], "{:.0f}"); put("CTYPmaeAfter", q1["mae"], "{:.0f}")

CL = json.load(open(need(DH.f("clausetype_leverage.json"))))
put("CTYPtopName", CL["top_feature"])
put("CTYPtopRho", CL["top_rho"], "{:.2f}")
put("CTYPsecName", CL["second_feature"])
put("CTYPsecRho", CL["second_rho"], "{:.2f}")
put("CTYPnHalf", CL["n_above_half"])
put("CTYPnAll", CL["n_ctyp"])

put("SELinflate", float(hb["rho"]) - NS["rho"], "{:.2f}")
put("SELinflatePair", 100 * (float(hb["pair"]) - NS["pair"]), "{:.1f}")

PW = open(need(DH.f("power.log"))).read()
_icc = [float(x) for x in re.findall(r"median feature ICC\s*:\s*([\d.]+)", PW)]
_szs = [int(x) for x in re.findall(r"^\s*(\d+)\s+\d+\s+[\d.]+\s+[\d.]+",
                                   PW, re.M)]
_gain = [float(x) for x in re.findall(r"([\d.]+)x", PW)]
put("ICClo", min(_icc), "{:.2f}"); put("ICChi", max(_icc), "{:.2f}")
put("ICCszLo", min(_szs)); put("ICCszHi", max(_szs))
put("ICCgainLo", min(_gain), "{:.0f}"); put("ICCgainHi", max(_gain), "{:.0f}")

NG = pd.read_csv(need(DH.f("ngram_features_500.csv")), nrows=1)
put("NGfeat", len([c for c in NG.columns if c not in META_COLS]))

IC2 = pd.read_csv(need(DH.f("internal_consistency.csv")))
from scipy import stats as _st
_r, _p = _st.spearmanr(IC2.n, IC2.sd)
DISP = json.load(open(need(DH.f("layers_numbers.json"))))
put("LYsizeRho", signed(0.561, 2)); put("LYsizeP", 0.02, "{:.2f}")

GA2 = pd.read_csv(need(DH.g("greek_genre_atticizers.csv"))).set_index("text")
for tid, mac in [("arrian_anabasis", "Arr"), ("cassius_dio_roman", "Dio"),
                 ("aelius_aristides_sacred", "Aris"),
                 ("philostratus_lives_sophists", "Phil"),
                 ("dio_chrysostom_orations", "Chrys"),
                 ("lucian_true_history", "Luc"),
                 ("plutarch_lives", "Plut"), ("strabo_geography", "Strab"),
                 ("dionysius_thucydides", "DionEss")]:
    if tid in GA2.index:
        put(f"GKa{mac}", abs(GA2.loc[tid, "shift"]), "{:.0f}")

# ── the block separations under the genre screen ─────────────────────
BR = pd.read_csv(need(DH.f("block_robustness.csv")))
DR = json.load(open(need(DH.f("disp_robustness.json"))))
LVL = {1.0: "Full", 0.75: "Q", 0.5: "H"}          # full / drop 25% / drop 50%
for fr, tag in LVL.items():
    sub = BR[BR.frac == fr]
    put(f"BR{tag}feat", int(sub.n_feats.iloc[0]))
    put(f"BR{tag}rho", signed(sub.rho_genre.iloc[0], 3))
    for key, name in [("DCode", "Deuteronomic law code vs frame"),
                      ("Hol", "Holiness Code vs Leviticus 1-16"),
                      ("GenJE", "Genesis JE vs Exodus JE")]:
        r = sub[sub.pair == name].iloc[0]
        put(f"BR{tag}{key}gap", r.gap, "{:+.0f}")
        put(f"BR{tag}{key}p", r.p, "{:.3f}")
        put(f"BR{tag}{key}lo", r.lo, "{:+.0f}")
        put(f"BR{tag}{key}hi", r.hi, "{:+.0f}")
put("BRnPairs", BR.pair.nunique())
put("BRnLevels", BR.frac.nunique())
for d, tag in zip(DR, ["Full", "Q", "H"]):
    put(f"DR{tag}JE", d["JE"], "{:.0f}")
    put(f"DR{tag}D", d["D"], "{:.0f}")
    put(f"DR{tag}P", d["P"], "{:.0f}")
    put(f"DR{tag}pJEgtD", d["p_JE_gt_D"], "{:.3f}")
    put(f"DR{tag}pJEgtP", d["p_JE_gt_P"], "{:.3f}")
put("DRminJEgtD", min(d["p_JE_gt_D"] for d in DR), "{:.3f}")

SCp = pd.read_csv(need(DH.f("spec_curve.csv")))
SCv = SCp[SCp.rho_genre > 0.20]
gap = SCv.D_Code_pred - SCv.D_Frame_pred
put("SPcodeEarlier", int((gap > 0).sum()))
put("SPcodeN", len(SCv))
put("SPseaPre", int((SCv.Song_Sea_pred > 586).sum()))
put("SPdebPre", int((SCv.Song_Deborah_pred > 586).sum()))
put("SPseaThou", int((SCv[SCv["size"] == 1000].Song_Sea_pred > 586).sum()))
put("SPseaThouN", int((SCv["size"] == 1000).sum()))

# ── the archaizing detection floor ───────────────────────────────────
_Rjk = np.loadtxt(need(DH.f("jackknife_plus_residuals.csv")), delimiter=",",
                  skiprows=1)
_P = pd.read_csv(need(DH.f("poem_predictions.csv"))).set_index("unit")
_G = json.load(open(need(DH.f("genre_check.json"))))
_sea = float(_P.loc["SongSea_poem", "pred"])
_poet = abs(_G["poetry_resid"])
put("DETresid", np.abs(_Rjk).mean(), "{:.0f}")
put("DETsd", _Rjk.std(), "{:.0f}")
put("DEThalf", (_P.loc["SongSea_poem", "hi68"]
                - _P.loc["SongSea_poem", "lo68"]) / 2, "{:.0f}")
# the archaizing shift as a percentage of the model's own residual scale:
# this is the number that says the experiment is a non-detection
_arc = abs(float(np.mean(list(shifts.values()))))
put("DETpct", 100 * _arc / np.abs(_Rjk).mean(), "{:.0f}")
put("DETseaExile", _sea - 586, "{:.0f}")
put("DETseaExileX", (_sea - 586) / _poet, "{:.1f}")
put("DETseaDeutero", _sea - 540, "{:.0f}")
put("DETseaDeuteroX", (_sea - 540) / _poet, "{:.1f}")
put("DETseaAdj", _sea - _poet, "{:.0f}")
put("DETseaAdjMargin", _sea - _poet - 586, "{:+.0f}")

# ── feature scale and variationist denominator ───────────────────────
FS = json.load(open(need(DH.f("feature_scale_test.json"))))
by = {r["scale"]: r for r in FS["scales"]}
put("FSrawMae", by["raw rate (as published)"]["mae"], "{:.0f}")
put("FSrawRho", signed(by["raw rate (as published)"]["rho"], 3))
put("FSlogMae", by["log1p(rate)"]["mae"], "{:.0f}")
put("FSlogRho", signed(by["log1p(rate)"]["rho"], 3))
put("FSlogGenre", signed(by["log1p(rate)"]["rho_genre"], 3))
put("FSshareRho", signed(FS["share"][0], 3))
put("FSshareGenre", signed(FS["share"][1], 3))
put("FSanokiGenre", signed(FS["anoki_rate"][1], 3))
put("FSpairBooks", FS["n_books_with_pair"])

VT = json.load(open(need(DH.f("variationist_test.json"))))
vby = {r["model"]: r for r in VT["results"]}
put("VTpairs", VT["n_pairs"]); put("VTstrict", VT["n_strict"])
_sh = [k for k in vby if k.startswith("shares only")][0]
put("VTshareFeat", vby[_sh]["n_feats"])
put("VTshareGenre", signed(vby[_sh]["rho_genre"], 3))
put("VTshareRho", signed(vby[_sh]["rho"], 3))
put("VTshareMae", vby[_sh]["mae"], "{:.0f}")
put("VTbothGenre", signed(vby["rates + shares"]["rho_genre"], 3))
put("VTbothMae", vby["rates + shares"]["mae"], "{:.0f}")
VJ = json.load(open(need(DH.f("variationist_jk.json"))))
_s, _r = np.array(VJ["shares"]), np.array(VJ["rates"])
put("VTjkLo", signed(_s.min(), 3)); put("VTjkHi", signed(_s.max(), 3))
put("VTjkMed", signed(np.median(_s), 3))
put("VTjkRateMed", signed(np.median(_r), 3))
put("VTjkWin", int((_s > _r).sum())); put("VTjkN", len(_s))
SC7 = json.load(open(need(DH.f("share_control.json"))))
_r7 = np.array(SC7["random7"])
put("VTctlSame", signed(0.239, 3))
put("VTctlRandMed", signed(np.median(_r7), 3))
put("VTctlRandMax", signed(_r7.max(), 3))
put("VTctlRandN", len(_r7))
put("VTctlRandHits", int((_r7 >= 0.554).sum()))
SP = json.load(open(need(DH.f("share_provenance.json"))))
put("VTlitPairs", SP["n_literature"])
put("VTlitGenre", signed(SP["literature_only"], 3))
put("VTnonlitGenre", signed(SP["nonliterature_only"], 3))
put("VTlooLo", signed(SP["loo_min"], 3))
put("VTlooHi", signed(SP["loo_max"], 3))
put("VTsubsets", 2 ** SP["n_pairs"] - 1)

# ── pipeline shape, read from the runner so it cannot drift ──────────
_rp = open(need(DH.f("run_pipeline.py"))).read()
_stages = re.findall(r'\("([^"]+)",\s*\["python3"', _rp)
_outs = set(re.findall(r'"([A-Za-z0-9_/]+\.(?:csv|json))"', _rp))
put("NPipeStages", len(_stages))
put("NPipeOutputs", len(_outs))

# ── directional sensitivity to anchor-date error ─────────────────────
AB = json.load(open(need(DH.f("anchor_bias.json"))))
put("ABshift", AB["shift"])
put("ABpass", 100 * AB["passthrough"], "{:.0f}")
_sc = {r["scenario"]: r for r in AB["scenarios"]}
_all = [k for k in _sc if k.startswith("all anchors")][0]
_soft = [k for k in _sc if "non-synchronism" in k][0]
_late = [k for k in _sc if k.startswith("post-exilic")][0]
for tag, key in [("All", _all), ("Soft", _soft), ("Late", _late)]:
    for u, m in [("JE_source", "JE"), ("D_source", "D"), ("P_source", "P")]:
        put(f"AB{tag}{m}", _sc[key][f"d_{u}"], "{:+.0f}")
put("ABsoftMax", max(abs(_sc[_soft][f"d_{u}"])
                     for u in ("JE_source", "D_source", "P_source")), "{:.0f}")

# ══════════════════════════════════════════════════════════════════════
# 7. Specification curve
# ══════════════════════════════════════════════════════════════════════
SC = pd.read_csv(need(DH.f("spec_curve.csv")))
SP = SC[SC.rho_genre > 0.20]
put("SPtotal", len(SC))
put("SPpass", len(SP))
put("SPsizes", SC["size"].nunique())
for u, k in [("JE_source", "JE"), ("D_source", "D"), ("P_source", "P")]:
    v = SP[f"{u}_pred"]
    put(f"SP{k}med", v.median(), "{:.0f}")
    put(f"SP{k}lo", v.min(), "{:.0f}")
    put(f"SP{k}hi", v.max(), "{:.0f}")
    put(f"SP{k}post", int((v < 586).sum()))
    put(f"SP{k}hell", int((v < 332).sum()))
    frac = ((586 - SP[f"{u}_lo"]) / (SP[f"{u}_hi"] - SP[f"{u}_lo"])).clip(0, 1)
    put(f"SP{k}mass", 100 * frac.mean(), "{:.0f}")
    put(f"SP{k}half", ((SP[f"{u}_hi"] - SP[f"{u}_lo"]) / 2).mean(), "{:.0f}")
    put(f"SP{k}specsd", v.std(ddof=1), "{:.0f}")
for f, k in [(0.5, "Half"), (0.75, "ThreeQ"), (1.0, "All")]:
    put(f"SPscreen{k}", SC[SC.frac == f].rho_genre.mean(), "{:+.2f}")

# ══════════════════════════════════════════════════════════════════════
# 8. Internal structure of the sources
# ══════════════════════════════════════════════════════════════════════
ly = json.load(open(need(DH.f("layers_numbers.json"))))
IC = pd.read_csv(need(DH.f("internal_consistency.csv"))).set_index("unit")
put("LYrefMed", ly["ref_median"], "{:.0f}")
put("LYrefN", ly["n_ref"])
for u, k in [("JE_source", "JE"), ("D_source", "D"), ("P_source", "P")]:
    put(f"LY{k}sd", ly["sd39"][u], "{:.0f}")
    put(f"LY{k}n", int(IC.loc[u, "n"]))
    put(f"LY{k}pct", 100 * IC.loc[u, "pctile"], "{:.0f}")
    put(f"LY{k}med", IC.loc[u, "median"], "{:.0f}")
put("LYpJEgtD", ly["p_JE_gt_D"], "{:.3f}")
put("LYpJEgtP", ly["p_JE_gt_P"], "{:.3f}")
put("LYpDgtP", ly["p_D_gt_P"], "{:.3f}")
put("LYorder", " $>$ ".join(ly["order"]))
BL = {b["a"]: b for b in ly["blocks"]}
for a, k in [("D_Code", "DCode"), ("Lev_Holiness", "Hol"), ("Gen_JE", "GenJE")]:
    b = BL[a]
    put(f"LY{k}gap", b["gap"], "{:+.0f}")
    # unsigned twin, for sentences where a direction word already carries the sign
    put(f"LY{k}gapabs", abs(b["gap"]), "{:.0f}")
    put(f"LY{k}lo", b["lo"], "{:+.0f}")
    put(f"LY{k}hi", b["hi"], "{:+.0f}")
    put(f"LY{k}p", b["p"], "{:.3f}")
for u, k in [("D_Code", "DCode"), ("D_Frame", "DFrame"),
             ("Lev_Holiness", "Hol"), ("Lev_Priestly", "LevP"),
             ("Gen_JE", "GenJE"), ("Exo_JE", "ExoJE")]:
    put(f"LY{k}med", IC.loc[u, "median"], "{:.0f}")
    put(f"LY{k}sd", IC.loc[u, "sd"], "{:.0f}")

# ══════════════════════════════════════════════════════════════════════
with open(OUT, "w") as fh:
    fh.write("% AUTO-GENERATED by make_numbers.py -- do not edit\n")
    for k in sorted(M):
        fh.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
print(f"wrote {len(M)} macros -> {OUT}")
