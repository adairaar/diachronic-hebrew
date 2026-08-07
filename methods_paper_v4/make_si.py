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

print(f"wrote S1, S2, S3, S4 ({len(FF)} features), S5 ({len(MAN)} texts), S6")
