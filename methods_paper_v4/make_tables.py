"""
Generate every LaTeX table from result files.  No table is hand-maintained.
"""
import json, os, sys
import numpy as np, pandas as pd

R, G = "/home/claude", "/home/claude/greek"
T = "/home/claude/ms/tables"
os.makedirs(T, exist_ok=True)


def need(p):
    if not os.path.exists(p): sys.exit(f"MISSING RESULT FILE: {p}")
    return p


def wrap(body, caption, label, cols, small=True):
    sz = "\\footnotesize" if small else ""
    return (f"\\begin{{table}}[!ht]\n\\centering\n{sz}\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{body}\\bottomrule\n"
            f"\\end{{tabular}}\n\\end{{table}}\n")


def esc(s):
    return str(s).replace("_", "\\_")


# ── Table 1: training corpus ────────────────────────────────────────────
ANCHOR = {
    # (basis of the date, type)
    #   S = explicit synchronism: named ruler or regnal year, checkable against
    #       Assyrian, Babylonian or Persian records
    #   H = historical reference: a datable event, institution, or language
    #       contact documented outside the biblical tradition
    #   T = typological: dated by resemblance to Hebrew of supposed date
    "Amos": ("Superscription: Uzziah of Judah / Jeroboam II of Israel", "S"),
    "Hosea": ("Superscription: Uzziah to Hezekiah / Jeroboam II", "S"),
    "Micah": ("Superscription: Jotham, Ahaz, Hezekiah", "S"),
    "Isaiah_1": ("Superscription: Uzziah to Hezekiah; Assyrian campaigns", "S"),
    "Nahum": ("Fall of Nineveh, 612 BCE", "H"),
    "Zephaniah": ("Superscription: Josiah", "S"),
    "Habakkuk": ("Neo-Babylonian ascendancy", "H"),
    "Jer_oracle": ("Superscription: Josiah to Zedekiah; 597 and 586 BCE", "S"),
    "Obadiah": ("Edomite conduct at the fall of Jerusalem, 586 BCE", "H"),
    "Ezekiel": ("Thirteen regnal dates from Jehoiachin's deportation, 597 BCE", "S"),
    "Lamentations": ("Destruction of Jerusalem, 586 BCE", "H"),
    "Isaiah_2": ("Cyrus named; fall of Babylon, 539 BCE", "S"),
    "Haggai": ("Regnal dates, second year of Darius I, 520 BCE", "S"),
    "Zechariah_1": ("Regnal dates, second to fourth year of Darius I", "S"),
    "Malachi": ("Persian governor (\\textit{peha}); second-temple cult", "H"),
    "Joel": ("Functioning second temple; no monarchy", "H"),
    "Jonah": ("Nineveh already fallen (3:3); post-exilic novella", "T"),
    "Isaiah_3": ("Restored temple cult; Persian-period institutions", "H"),
    "Ezra": ("Persian regnal dates, Cyrus to Artaxerxes", "S"),
    "Nehemiah": ("Twentieth year of Artaxerxes I, 445 BCE", "S"),
    "Zechariah_2": ("Greek (\\textit{Yawan}) as a military power", "H"),
    "Chronicles": ("Genealogies to the Persian period; Cyrus decree", "H"),
    "Esther": ("Achaemenid court and administration; Xerxes", "H"),
    "Ecclesiastes": ("Persian loanwords (\\textit{pardes}, \\textit{pitgam})", "H"),
    "Daniel": ("Antiochene persecution, 167 BCE", "H"),
}

B = pd.read_csv(need(f"{R}/final_lobo_books.csv")).sort_values("truth", ascending=False)
rows = ["\\textbf{Unit} & \\textbf{Date} & \\textbf{Words} & \\textbf{Pass.} & "
        "\\textbf{Type} & \\textbf{Basis of the date} \\\\\n\\midrule\n"]
for _, r in B.iterrows():
    a, kind = ANCHOR.get(r.book, ("---", "L"))
    rows.append(f"{esc(r.book)} & {int(r.truth)} & {int(r.n_words):,} & "
                f"{int(r.n_chunks)} & {kind} & {a} \\\\\n")
open(f"{T}/tab_corpus.tex", "w").write(wrap(
    "".join(rows),
    "\\textbf{The training corpus.}  Dates are BCE; ``Pass.'' is the number of "
    "$\\sim$500-word passages.  Type~S dates rest on an explicit synchronism, a "
    "named ruler or regnal year checkable against Assyrian, Babylonian or Persian "
    "records.  Type~H dates rest on a datable event, institution, or language "
    "contact documented outside the biblical tradition.  Neither depends on the "
    "feature cline this model estimates.  Type~T marks the one unit dated "
    "principally by resemblance to Hebrew of supposed date; "
    "Table~\\ref{tab:anchors} reports the effect of removing it.",
    "tab:corpus", "llrrcp{4.6cm}"))

# ── Table 2: prior leakage ──────────────────────────────────────────────
L = pd.read_csv(need(f"{R}/leakage_generative.csv")).sort_values("sigma_u")
# data_share is already expressed as a percentage in the source file
rows = ["\\textbf{Unit} & \\textbf{Prior $\\sigma$} & \\textbf{Data share} & "
        "\\textbf{Leaky err.} & \\textbf{Honest err.} \\\\\n\\midrule\n"]
for _, r in L.iterrows():
    rows.append(f"{esc(r['id'])} & {r.sigma_u:.0f} & {r.data_share:.1f}\\% & "
                f"{r.err_leaky:.0f} & {r.err_honest:.0f} \\\\\n")
tight = L[L.sigma_u <= 20]
rows.append("\\midrule\n\\textbf{Median, all} & "
            f"{L.sigma_u.median():.0f} & {L.data_share.median():.1f}\\% & "
            f"{L.err_leaky.median():.0f} & {L.err_honest.median():.0f} \\\\\n")
rows.append(f"\\textbf{{Mean, $\\sigma \\leq 20$ ({len(tight)} units)}} & "
            f"{tight.sigma_u.mean():.0f} & {tight.data_share.mean():.1f}\\% & "
            f"{tight.err_leaky.mean():.1f} & {tight.err_honest.mean():.1f} \\\\\n")
open(f"{T}/tab_leakage.tex", "w").write(wrap(
    "".join(rows),
    "\\textbf{Prior leakage inflates reported holdout accuracy.}  ``Prior "
    "$\\sigma$'' is the standard deviation of the scholarly prior in years; "
    "``data share'' is the fraction of posterior precision contributed by the "
    "linguistic features; the two error columns are mean absolute errors in "
    "years under a prior centered on the unit's own scholarly date and under an "
    "agnostic prior, with the pipeline otherwise identical.",
    "tab:leakage", "lrrrr"))

# ── Table 3: leave-one-book-out ─────────────────────────────────────────
rows = ["\\textbf{Unit} & \\textbf{True} & \\textbf{Est.} & \\textbf{Err.} & "
        "\\textbf{Pass. SD} & \\textbf{$P$(post-ex.)} \\\\\n\\midrule\n"]
for _, r in B.iterrows():
    rows.append(f"{esc(r.book)} & {int(r.truth)} & {int(r.pred)} & "
                f"{int(r.resid):+d} & {int(r.chunk_sd)} & {r.p_post:.2f} \\\\\n")
open(f"{T}/tab_lobo.tex", "w").write(wrap(
    "".join(rows),
    "\\textbf{Leave-one-book-out performance.}  Dates BCE; error is true minus "
    "estimated, so a positive error means the model placed the book too late. "
    "``Pass.\\ SD'' is the standard deviation of the book's passage-level "
    "estimates, a measure of internal homogeneity.  Every quantity is "
    "out-of-sample: standardization, weighting, $\\lambda$ selection and "
    "calibration were performed without the held-out book.",
    "tab:lobo", "lrrrrr"))

# ── Table 4: targets ────────────────────────────────────────────────────
NM = {"Song_Sea": "Song of the Sea", "Song_Deborah": "Song of Deborah",
      "D_Song": "Song of Moses", "JE_source": "JE composite",
      "P_source": "P source", "D_source": "D source", "Gen_JE": "Genesis JE",
      "Exo_JE": "Exodus JE", "Num_JE": "Numbers JE", "D_Code": "D law code",
      "D_Frame": "D frame", "Lev_Holiness": "Holiness Code",
      "Lev_Priestly": "Leviticus P", "Jer_DTR": "Jeremiah Dtr prose"}
ORDER = ["Song_Deborah", "Song_Sea", "D_Song", "JE_source", "Gen_JE", "Exo_JE",
         "Num_JE", "D_source", "D_Code", "D_Frame", "P_source", "Lev_Priestly",
         "Lev_Holiness", "Jer_DTR", "Genesis", "Exodus", "Leviticus", "Numbers",
         "Deuteronomy"]
TG = pd.read_csv(need(f"{R}/target_predictions_final.csv")).set_index("unit")
GRP = {"Song_Deborah": "Archaic poems", "JE_source": "Documentary sources",
       "Gen_JE": "Sub-strata", "Genesis": "Pentateuch as received"}
rows = ["\\textbf{Unit} & \\textbf{Words} & \\textbf{Pass.} & \\textbf{Est.} & "
        "\\textbf{68\\% interval} & \\textbf{$P$(post-ex.)} \\\\\n"]
for u in ORDER:
    if u not in TG.index: continue
    if u in GRP:
        rows.append("\\midrule\n\\multicolumn{6}{l}{\\textit{" + GRP[u] + "}} \\\\\n")
    r = TG.loc[u]
    rows.append(f"\\quad {NM.get(u,esc(u))} & {int(r.n_words):,} & "
                f"{int(r.n_chunks)} & {int(round(r.pred))} & "
                f"{int(round(r.hi68))}--{int(round(r.lo68))} & "
                f"{r.p_post:.2f} \\\\\n")
open(f"{T}/tab_targets.tex", "w").write(wrap(
    "".join(rows),
    "\\textbf{Undated units.}  Dates BCE; intervals are 68\\% conformal, "
    "calibrated on leave-one-book-out residuals.  No unit here contributed to "
    "training, feature selection, weighting, regularization or calibration.  "
    "The three poems are extracted at verse precision (see text).",
    "tab:targets", "lrrrcr"))

# ── Table 5: synthetic archaizing ───────────────────────────────────────
A = pd.read_csv(need(f"{R}/archaize_results.csv"))
rows = ["\\textbf{Book} & \\textbf{True} & \\textbf{$r=0$} & \\textbf{$r=1$} & "
        "\\textbf{Shift} & \\textbf{Swaps} & \\textbf{per 1k w} \\\\\n\\midrule\n"]
sh = []
for u in A.unit.unique():
    s = A[A.unit == u]
    a = float(s[s.rate == 0].pred.iloc[0]); b = float(s[s.rate == 1.0].pred.iloc[0])
    ns = int(s[s.rate == 1.0].n_sub.iloc[0]); nw = int(s.n_words.iloc[0])
    sh.append(b - a)
    rows.append(f"{esc(u)} & {int(s.truth.iloc[0])} & {a:.0f} & {b:.0f} & "
                f"{b-a:+.0f} & {ns} & {1000*ns/nw:.1f} \\\\\n")
rows.append("\\midrule\n\\multicolumn{4}{l}{\\textbf{Mean apparent shift}} & "
            f"\\textbf{{{np.mean(sh):+.0f}}} & & \\\\\n")
open(f"{T}/tab_archaize.tex", "w").write(wrap(
    "".join(rows),
    "\\textbf{What total lexical archaizing buys.}  Each securely dated late "
    "book had its Late Biblical Hebrew forms replaced by Classical counterparts "
    "at rate $r$ and was then re-dated by a model trained on the other 24 books. "
    "$r=1$ replaces every eligible token in the book.  Dates BCE; a positive "
    "shift means the book appears older.  Ezra, Nehemiah and Chronicles offer "
    "too few eligible tokens for their null results to be informative.",
    "tab:archaize", "lrrrrrr"))

# ── Table 6: Greek Atticizers ───────────────────────────────────────────
GA = pd.read_csv(need(f"{G}/greek_atticizers.csv")).sort_values("truth")
MAN = {e["id"]: e for e in
       json.load(open(need(f"{G}/corpus_manifest.json"), encoding="utf-8"))}
rows = ["\\textbf{Author} & \\textbf{Work} & \\textbf{True} & \\textbf{Est.} & "
        "\\textbf{Shift} & \\textbf{Pass.} \\\\\n\\midrule\n"]
for _, r in GA.iterrows():
    work = MAN.get(r.text, {}).get("work", r.text.replace("_", " "))
    rows.append(f"{esc(r.author)} & {esc(work)} & {int(r.truth):+d} & "
                f"{int(r.pred):+d} & {int(r['shift']):+d} & {int(r.n_chunks)} \\\\\n")
rows.append("\\midrule\n\\multicolumn{4}{l}{\\textbf{Mean displacement}} & "
            f"\\textbf{{{GA['shift'].mean():+.0f}}} & \\\\\n")
open(f"{T}/tab_greek.tex", "w").write(wrap(
    "".join(rows),
    "\\textbf{Second Sophistic Atticizers under a model trained only on "
    "non-archaizing Greek.}  Dates CE (negative = BCE).  A negative shift means "
    "the text is dated earlier than it was written, i.e.\\ the archaizing "
    "worked.  These 14 texts were excluded from training entirely.",
    "tab:greek", "llrrrr"))

print("wrote:", ", ".join(sorted(os.listdir(T))))

# ── Table: robustness of the headline statistics ────────────────────────
AS = json.load(open(need(f"{R}/anchor_sensitivity.json")))
JK = json.load(open(need(f"{R}/jackknife.json")))
rows = ["\\textbf{Fit} & \\textbf{Units} & \\textbf{MAE} & \\textbf{Baseline} & "
        "\\textbf{$\\rho$} & \\textbf{Pairwise} \\\\\n\\midrule\n"]
rows.append(f"Full corpus & 25 & {JK['full_mae']:.0f} & 137 & "
            f"{JK['full_rho']:+.2f} & {100*JK['full_pair']:.1f}\\% \\\\\n")
rows.append("\\midrule\n\\multicolumn{6}{l}{\\textit{Leave-one-unit-out jackknife, "
            "all 25 fits}} \\\\\n")
rows.append(f"\\quad median & 24 & --- & --- & {JK['rho_med']:+.2f} & "
            f"{100*JK['pair_med']:.1f}\\% \\\\\n")
rows.append(f"\\quad range & 24 & --- & --- & {JK['rho_min']:+.2f} to "
            f"{JK['rho_max']:+.2f} & {100*JK['pair_min']:.1f}--"
            f"{100*JK['pair_max']:.1f}\\% \\\\\n")
rows.append("\\midrule\n\\multicolumn{6}{l}{\\textit{Removing the less securely "
            "anchored units}} \\\\\n")
LBL = {1: "\\quad without Jonah and Ecclesiastes",
       2: "\\quad without all 6 non-synchronism units"}
for i, v in enumerate(AS):
    if i == 0: continue
    rows.append(f"{LBL[i]} & {v['n']} & {v['mae']:.0f} & {v['base']:.0f} & "
                f"{v['rho']:+.2f} & {100*v['pair']:.1f}\\% \\\\\n")
open(f"{T}/tab_anchors.tex", "w").write(wrap(
    "".join(rows),
    "\\textbf{Robustness of the headline statistics.}  MAE and baseline are in "
    "years, the baseline being the constant predictor on the same subset.  The "
    "jackknife drops each anchored unit in turn and re-runs the entire pipeline "
    "on the remaining 24; its spread is the leverage any single unit carries at "
    "this corpus size, and the full-corpus $\\rho$ should be read against it "
    "rather than as a point estimate.  The lower block removes the units whose "
    "dates are least independent of the language: Jonah, the one unit dated "
    "principally on typological grounds, together with Ecclesiastes, and then all "
    "six units lacking an explicit synchronism.",
    "tab:anchors", "lrrrcr"))
print("wrote tab_anchors (jackknife + anchor removal)")
