"""
Supporting-information tables, generated from the same specs the pipeline used.

S1 is emitted from the TARGETS dict in target_chunks.py rather than retyped, so
the appendix cannot drift from the partition that was actually analysed.
"""
import os, sys, json, re, ast
import numpy as np, pandas as pd

R, G = "/home/claude", "/home/claude/greek"
T = "/home/claude/ms/tables"
os.makedirs(T, exist_ok=True)


def need(p):
    if not os.path.exists(p): sys.exit(f"MISSING RESULT FILE: {p}")
    return p


def esc(s):
    return str(s).replace("_", "\\_").replace("&", "\\&")


def tt(s):
    """Escape an ETCBC feature name for \\texttt.  The transliteration uses
    <, > and [ as consonant signs, and OT1 would silently turn the angles into
    inverted punctuation."""
    return (str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_")
            .replace("&", "\\&").replace("#", "\\#").replace("%", "\\%")
            .replace("$", "\\$").replace("<", "\\textless{}")
            .replace(">", "\\textgreater{}").replace("^", "\\^{}")
            .replace("~", "\\~{}"))


# ── S1: source partitions, read out of the pipeline source ──────────────
src = open(need(f"{R}/target_chunks.py")).read()
m = re.search(r"TARGETS\s*=\s*(\{.*?\n\})", src, re.S)
if not m: sys.exit("could not locate TARGETS in target_chunks.py")
TARGETS = ast.literal_eval(m.group(1))

BOOK = {"Numeri": "Numbers", "Deuteronomium": "Deuteronomy", "Judices": "Judges",
        "Jeremia": "Jeremiah", "Threni": "Lamentations"}
NM = {"P_source": "P source", "JE_source": "JE composite", "D_source": "D source",
      "Gen_JE": "Genesis JE", "Exo_JE": "Exodus JE", "Num_JE": "Numbers JE",
      "D_Code": "D law code", "D_Frame": "D frame", "D_Song": "Song of Moses",
      "Lev_Holiness": "Holiness Code", "Lev_Priestly": "Leviticus P",
      "Song_Sea": "Song of the Sea", "Song_Deborah": "Song of Deborah",
      "Jer_DTR": "Jeremiah Dtr prose"}
POEM_NOTE = {"Song_Sea": "Exod 15:1--18 at verse precision (see text)",
             "Song_Deborah": "Judg 5:2--31 at verse precision",
             "D_Song": "Deut 32:1--43 at verse precision"}

rows = []
for uid, spec in TARGETS.items():
    if uid in ("Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy"):
        continue
    parts = []
    for book, rngs in spec:
        bn = BOOK.get(book, book)
        rs = ", ".join(f"{a}" if a == b else f"{a}--{b}" for a, b in rngs)
        parts.append(f"{bn} {rs}")
    note = POEM_NOTE.get(uid, "")
    rows.append((NM.get(uid, uid), "; ".join(parts), note))

body = ["\\textbf{Unit} & \\textbf{Chapters} \\\\\n\\midrule\n"]
for name, chapters, note in rows:
    body.append(f"{esc(name)} & {esc(chapters)}"
                + (f" \\newline \\textit{{{note}}}" if note else "") + " \\\\\n")
open(f"{T}/tab_s1_partitions.tex", "w").write(
    "\\begin{table}[!ht]\n\\centering\n\\footnotesize\n"
    "\\caption{{\\bf S1 Appendix. Source partitions.}  Chapter specification of "
    "every undated unit.  The five Pentateuchal books as received are omitted "
    "(they are simply the whole book).  Partitions follow the standard critical "
    "division; the three poems were additionally re-extracted at verse precision.}\n"
    "\\label{tab:s1}\n\\begin{tabular}{lp{9.2cm}}\n\\toprule\n"
    + "".join(body) + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ── S2: power analysis ──────────────────────────────────────────────────
pw = open(need(f"{R}/power.log")).read()
blocks = re.findall(r"chunk\s+chunks\s+ICC\s+DEFF\s+n_eff\s+gain vs n=25\n((?:.+\n)+)", pw)
body = ["\\textbf{Passage size} & \\textbf{Passages} & \\textbf{Median ICC} & "
        "\\textbf{Design effect} & \\textbf{$n_{\\mathrm{eff}}$} & "
        "\\textbf{Gain vs $n=25$} \\\\\n\\midrule\n"]
if blocks:
    for line in blocks[-1].strip().splitlines():
        p = line.split()
        if len(p) >= 6:
            body.append(" & ".join(esc(x) for x in p[:6]) + " \\\\\n")
open(f"{T}/tab_s2_power.tex", "w").write(
    "\\begin{table}[!ht]\n\\centering\n\\footnotesize\n"
    "\\caption{{\\bf S2 Table. Effective sample size by passage size.}  "
    "Passages within a book are correlated, so the raw passage count overstates "
    "the information available.  $n_{\\mathrm{eff}} = n/(1+(\\bar m -1)\\rho_I)$ "
    "with $\\rho_I$ the median intraclass correlation across features.}\n"
    "\\label{tab:s2}\n\\begin{tabular}{rrrrrr}\n\\toprule\n"
    + "".join(body) + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ── S3: configuration sweep ─────────────────────────────────────────────
SW = pd.read_csv(need(f"{R}/sweep_summary.csv"))
cols = [c for c in ["target", "kind", "p", "calib", "mae", "rho", "pair", "cov"]
        if c in SW.columns]
SW = SW[cols].sort_values(["target", "kind"])
hdr = {"target": "Passage", "kind": "Features", "p": "$p$", "calib": "Calib.",
       "mae": "MAE", "rho": "$\\rho$", "pair": "Pair \\%", "cov": "Coverage"}
body = [" & ".join(f"\\textbf{{{hdr.get(c,c)}}}" for c in cols) + " \\\\\n\\midrule\n"]
for _, r in SW.iterrows():
    cells = []
    for c in cols:
        v = r[c]
        if c == "mae": cells.append(f"{v:.0f}")
        elif c in ("rho", "cov"): cells.append(f"{v:+.2f}" if c == "rho" else f"{v:.2f}")
        elif c == "pair": cells.append(f"{100*v:.1f}")
        else: cells.append(esc(v))
    body.append(" & ".join(cells) + " \\\\\n")
open(f"{T}/tab_s3_sweep.tex", "w").write(
    "\\begin{table}[!ht]\n\\centering\n\\footnotesize\n"
    "\\caption{{\\bf S3 Table. Configuration search.}  Out-of-sample metrics for "
    "each combination of passage size, feature set and calibration under "
    "leave-one-book-out.  ``Coverage'' is the empirical coverage of the nominal "
    "68\\% interval; values near 1.0 indicate conservative intervals and values "
    "near 0.5 indicate the shrinkage problem discussed in the Model section.}\n"
    "\\label{tab:s3}\n\\begin{tabular}{" + "r" * len(cols) + "}\n\\toprule\n"
    + "".join(body) + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ── S6: Greek register variants ─────────────────────────────────────────
V = json.load(open(need(f"{G}/greek_register_variants.json")))
body = ["\\textbf{Training set} & \\textbf{Texts} & \\textbf{Tested} & "
        "\\textbf{MAE} & \\textbf{$\\rho$} & \\textbf{Mean shift} & "
        "\\textbf{Too early} \\\\\n\\midrule\n"]
for v in V:
    body.append(f"{esc(v['variant'])} & {v['n_train']} & {v['n_test']} & "
                f"{v['mae']:.0f} & {v['rho']:+.2f} & {v['mean_shift']:+.0f} & "
                f"{v['n_early']}/{v['n_test']} \\\\\n")
open(f"{T}/tab_s6_registers.tex", "w").write(
    "\\begin{table}[!ht]\n\\centering\n\\footnotesize\n"
    "\\caption{{\\bf S6 Table. Register sensitivity of the Greek archaizing "
    "measurement.}  The displacement of the Atticizers depends on which texts "
    "are treated as non-archaizing training material.  Variant A uses the corpus "
    "manifest's own labels; B additionally reclassifies authors independently "
    "described as classicizing in the literary-historical record; C discards "
    "register labels entirely and trains only on texts composed before 50~CE, "
    "prior to the Second Sophistic.  Dates CE; a negative shift means the text "
    "is dated earlier than it was written.}\n"
    "\\label{tab:s6}\n\\begin{tabular}{lrrrrrr}\n\\toprule\n"
    + "".join(body) + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ── S4: feature leverage ────────────────────────────────────────────────
FF = pd.read_csv(need(f"{R}/sensitivity_features.csv"))
FAM = pd.read_csv(need(f"{R}/sensitivity_families.csv"))
FF = FF.sort_values("abs", ascending=False).reset_index(drop=True)

fam_body = ["\\textbf{Family} & \\textbf{Features} & \\textbf{Total $|$leverage$|$} "
            "& \\textbf{Share} \\\\\n\\midrule\n"]
for _, r in FAM.iterrows():
    fam_body.append(f"{esc(r['family'])} & {int(r['count'])} & "
                    f"{r['total']:.0f} & {r['share']:.1f}\\% \\\\\n")
fam_tab = ("\\begin{table}[!ht]\n\\centering\n\\footnotesize\n"
           "\\caption{{\\bf S4 Table. Feature leverage by family.}  Leverage is the "
           "shift in apparent date, in years, produced by moving a feature one "
           "standard deviation while holding the rest of the passage fixed.  No "
           "family carries the signal alone and no single feature approaches the "
           "size of the effects being estimated; the full per-feature listing "
           "follows.}\n\\label{tab:s4}\n\\begin{tabular}{lrrr}\n\\toprule\n"
           + "".join(fam_body) + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")

NC = 3                                    # feature/leverage pairs across the page
rows = [FF.iloc[i:i + 1].iloc[0] for i in range(len(FF))]
nr = -(-len(rows) // NC)
lt = ["\\begingroup\\scriptsize\n",
      "\\setlength{\\tabcolsep}{3pt}\n",
      "\\begin{longtable}{" + "lr" * NC + "}\n",
      "\\caption{{\\bf S4 Table, continued. Per-feature leverage.}  All "
      f"{len(FF)} features, years of apparent date per standard-deviation shift, "
      "ordered by magnitude.  Read down the first pair of columns, then the "
      "second, then the third.}\n\\label{tab:s4full}\\\\\n\\toprule\n",
      " & ".join(["\\textbf{Feature} & \\textbf{yr/sd}"] * NC) + " \\\\\n\\midrule\n"
      "\\endfirsthead\n\\toprule\n"
      + " & ".join(["\\textbf{Feature} & \\textbf{yr/sd}"] * NC)
      + " \\\\\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n"]
for i in range(nr):
    cells = []
    for c in range(NC):
        k = c * nr + i
        if k < len(rows):
            cells += [f"\\texttt{{{tt(rows[k]['feature'])}}}",
                      f"{rows[k]['yr_per_sd']:+.2f}"]
        else:
            cells += ["", ""]
    lt.append(" & ".join(cells) + " \\\\\n")
lt.append("\\end{longtable}\n\\endgroup\n")
open(f"{T}/tab_s4_leverage.tex", "w").write(fam_tab + "\n" + "".join(lt))

# ── S5: the Greek corpus ────────────────────────────────────────────────
MAN = json.load(open(need(f"{G}/corpus_manifest.json")))
MAN = sorted(MAN, key=lambda t: t["date_ce"])
REG = {"ancient_Attic": "ancient Attic", "Atticizing": "Atticizing",
       "Koine": "Koine", "LXX": "LXX"}
S5HDR = ("\\textbf{Author} & \\textbf{Work} & \\textbf{Date} & "
         "\\textbf{Genre} & \\textbf{Register} & \\textbf{Tokens} "

         "\\\\\n\\midrule\n")
lt = ["\\begingroup\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n",
      "\\begin{longtable}{p{2.6cm}p{4.1cm}rll r}\n",
      "\\caption{{\\bf S5 Table. The Greek corpus.}  All "
      f"{len(MAN)} texts, with the composition date and its assumed standard "
      "deviation, the genre, the register label from the manifest, and the token "
      "count after extraction.  Dates are CE; negative values are BCE.  The "
      "fourteen texts labelled Atticizing are the archaizing authors whose "
      "displacement is measured in the Greek section; they are excluded from "
      "the training set under variant~A of Table~\\ref{tab:s6}.  A dagger marks "
      "the five texts reserved as a date-validation holdout."
      "}\n\\label{tab:s5}\\\\\n\\toprule\n",
      S5HDR + "\\endfirsthead\n\\toprule\n" + S5HDR
      + "\\endhead\n\\bottomrule\n\\endfoot\n"]
for t in MAN:
    w = esc(t["work"])
    if t.get("holdout"): w += " $^{\\dagger}$"
    ntok = f"{int(t.get('n_tokens', 0)):,}".replace(",", "\\,")
    lt.append(f"{esc(t['author'])} & {w} & "
              f"{int(t['date_ce'])}\\,$\\pm$\\,{int(t['date_sigma'])} & "
              f"{esc(t['genre'].replace('prose_', '').replace('_', ' '))} & "
              f"{esc(REG.get(t['register'], t['register']))} & "
              f"{ntok} \\\\\n")
lt.append("\\end{longtable}\n\\endgroup\n")
open(f"{T}/tab_s5_greek.tex", "w").write("".join(lt))

# ── S7: p-values as a function of where a permutation run is stopped ────
NS = pd.read_csv(need(f"{R}/null_stability.csv"))
NAME = {"within-genre": "Within-genre shuffle, genre-controlled $\\rho$",
        "free shuffle": "Free shuffle, raw $\\rho$"}
body = ["\\textbf{Null} & \\textbf{Draws} & \\textbf{$p$} \\\\\n\\midrule\n"]
for t, sub in NS.groupby("test", sort=False):
    first = True
    for _, r in sub.iterrows():
        lab = NAME[t] if first else ""
        body.append(f"{lab} & {int(r.draws)} & {r.p:.4f} \\\\\n")
        first = False
    body.append("\\addlinespace\n")
open(f"{T}/tab_s7_stopping.tex", "w").write(
    "\\begin{table}[!ht]\n\\centering\n\\footnotesize\n"
    "\\caption{{\\bf Sensitivity of each permutation $p$-value to the stopping "
    "point.}  Permutation runs are expensive, and a run stopped after its "
    "result is known invites the objection that it was stopped favourably.  "
    "Both nulls are therefore reported at a stopping point declared in advance, "
    "and this table gives the $p$-value that each prefix of the same draw "
    "sequence would have produced.  The within-genre $p$ falls below 0.05 "
    "before 100 draws and stays there; the free-shuffle $p$ is below 0.01 "
    "throughout.  The exact value drifts, but neither conclusion depends on "
    "where the run was stopped.}\n"
    "\\label{tab:s7}\n\\begin{tabular}{lrr}\n\\toprule\n"
    + "".join(body) + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ── S9: the full specification grid ─────────────────────────────────────
SC = pd.read_csv(need(f"{R}/spec_curve.csv")).sort_values("rho_genre",
                                                          ascending=False)
body = ["\\textbf{Words} & \\textbf{Keep} & \\textbf{Wt} & \\textbf{Cal} & "
        "\\textbf{$\\rho$} & \\textbf{$\\rho|$g} & \\textbf{MAE} & "
        "\\textbf{JE} & \\textbf{D} & \\textbf{P} \\\\\n\\midrule\n"]
for _, r in SC.iterrows():
    mark = "" if r.rho_genre > 0.20 else "$^{\\dagger}$"
    body.append(
        f"{int(r['size'])}{mark} & {int(100*r.frac)}\\% & "
        f"{'yes' if r.alpha else 'no'} & {'var' if r.calib=='var' else 'none'} & "
        f"{r.rho_raw:+.2f} & {r.rho_genre:+.2f} & {r.mae:.0f} & "
        f"{r.JE_source_pred:.0f} & {r.D_source_pred:.0f} & "
        f"{r.P_source_pred:.0f} \\\\\n")
open(f"{T}/tab_s9_speccurve.tex", "w").write(
    "\\begin{table}[!ht]\n\\centering\n\\scriptsize\n"
    "\\caption{{\\bf Every analytic specification.}  All "
    f"{len(SC)} combinations of passage size, feature screen (share of features "
    "retained after dropping the most genre-diagnostic), inverse-density "
    "weighting and variance-matched calibration.  Each row is a complete "
    "leave-one-book-out run.  $\\rho$ is raw Spearman correlation with true "
    "date, $\\rho|$g the same with genre held fixed, MAE the mean absolute "
    "error in years, and the last three columns the point estimates for the "
    "sources in BCE.  Rows marked $\\dagger$ fall below the "
    "$\\rho|\\mathrm{g} > 0.20$ validation threshold and are excluded from "
    "every figure quoted in the text.  Sorted by validation statistic.}\n"
    "\\label{tab:s9}\n\\begin{tabular}{rrllrrrrrr}\n\\toprule\n"
    + "".join(body) + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ── S10: the red-team ablations ─────────────────────────────────────────
RT = pd.read_csv(need(f"{R}/red_team.csv"))
body = ["\\textbf{Variant} & \\textbf{Features} & \\textbf{Books} & "
        "\\textbf{MAE} & \\textbf{$\\rho$} & \\textbf{$\\rho|$genre} & "
        "\\textbf{$\\rho$ proph} \\\\\n\\midrule\n"]
NAMES = {"A": "Baseline: all features, all anchors",
         "B": "Morphosyntax only (lexemes removed)",
         "C": "Lexemes only (morphosyntax removed)",
         "D": "Externally anchored books only",
         "E": "Morphosyntax only, external anchors only"}
for _, r in RT.iterrows():
    body.append(f"{NAMES[r.label[0]]} & {int(r.n_feats)} & {int(r.n_books)} & "
                f"{r.mae:.0f} & {r.rho_raw:+.3f} & {r.rho_genre:+.3f} & "
                f"{r.rho_proph:+.3f} \\\\\n")
open(f"{T}/tab_s10_redteam.tex", "w").write(
    "\\begin{table}[!ht]\n\\centering\n\\footnotesize\n"
    "\\caption{{\\bf Ablations designed to break the result.}  Each variant "
    "removes something the result might depend on and refits from scratch.  "
    "Comparing B against C isolates which feature family carries the signal; "
    "D and E remove the anchors whose dates rest on literary rather than "
    "external grounds.  Note that D also strips four late prophetic books and "
    "so restricts the within-genre chronological range, which cannot be "
    "separated from the circularity it is meant to test.}\n"
    "\\label{tab:s10}\n\\begin{tabular}{lrrrrrr}\n\\toprule\n"
    + "".join(body) + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ── S11: block separations under the genre screen ───────────────────────
BR = pd.read_csv(need(f"{R}/block_robustness.csv"))
DRj = json.load(open(need(f"{R}/disp_robustness.json")))
LAB = {1.0: "All features", 0.75: "Drop top 25\\%", 0.5: "Drop top 50\\%"}
body = ["\\textbf{Feature set} & \\textbf{$p$ feats} & "
        "\\textbf{$\\rho|$genre} & \\textbf{Comparison} & \\textbf{Gap} & "
        "\\textbf{95\\% CI} & \\textbf{$p$} \\\\\n\\midrule\n"]
for fr in (1.0, 0.75, 0.5):
    sub = BR[BR.frac == fr]
    first = True
    for _, r in sub.iterrows():
        lead = (f"{LAB[fr]} & {int(r.n_feats)} & {r.rho_genre:+.3f}"
                if first else " & & ")
        body.append(f"{lead} & {r['pair']} & {r.gap:+.0f} & "
                    f"{r.lo:+.0f} to {r.hi:+.0f} & {r.p:.3f} \\\\\n")
        first = False
    body.append("\\addlinespace\n")
open(f"{T}/tab_s11_blocks.tex", "w").write(
    "\\begin{table}[!ht]\n\\centering\n\\footnotesize\n"
    "\\caption{{\\bf Block separations under the genre screen.}  Difference in "
    "median passage estimate between the conventionally distinguished blocks of "
    "each source, recomputed with the most genre-diagnostic features removed.  "
    "A positive gap places the first-named block earlier.  Screening improves "
    "the model's genre-controlled ordering of the anchors (third column) while "
    "removing the separations entirely, and reversing the Deuteronomic one.  "
    "Widening intervals reflect the smaller feature set; the sign change in "
    "Deuteronomy does not.}\n"
    "\\label{tab:s11}\n\\begin{tabular}{lrrlrrr}\n\\toprule\n"
    + "".join(body) + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")

print(f"wrote S1, S2, S3, S4 ({len(FF)} features), S5 ({len(MAN)} texts), "
      f"S6, S7 ({len(NS)} stopping points), S9 ({len(SC)} specs), "
      f"S10 ({len(RT)} variants), S11 ({len(BR)} block tests)")
