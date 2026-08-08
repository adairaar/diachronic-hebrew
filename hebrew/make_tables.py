"""
Generate main.tex tables directly from corpus_manifest_v2.json and the
model outputs, so the manuscript can never again describe a corpus that
differs from the one the code uses.

Writes:
  tab_corpus.tex   — Table 1, rebuilt training corpus
  tab_targets.tex  — new results table, conformal date ranges
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import json, pandas as pd, numpy as np

man = json.load(open(DH.f("corpus_manifest_v2.json")))
fm = pd.read_csv(DH.f("feature_matrix_v2.csv"))
words = dict(zip(fm["id"], fm["n_words"]))

REG = {"SBH": "S", "Transitional": "T", "LBH": "L"}
LABEL = {  # manifest id -> manuscript display name
    "Isaiah_1": "Isaiah 1--39", "Isaiah_2": "Isaiah 40--55",
    "Isaiah_3": "Isaiah 56--66", "Zechariah_1": "Zechariah 1--8",
    "Zechariah_2": "Zechariah 9--14", "Daniel": "Daniel (Heb.)",
    "Jer_oracle": "Jeremiah oracle", "Jer_DTR": "Jeremiah Dtr",
    "Song_Sea": "Song of the Sea", "Song_Deborah": "Song of Deborah",
    "D_source": "D source", "P_source": "P source", "JE_source": "JE source",
    "D_Code": "D Code", "D_Frame": "D Frame", "D_Song": "D Song",
    "Lev_Holiness": "Holiness Code", "Lev_Priestly": "Leviticus P",
    "Gen_JE": "Genesis JE", "Exo_JE": "Exodus JE", "Num_JE": "Numbers JE",
    "Ecclesiastes": "Ecclesiastes", "Lamentations": "Lamentations",
}
def nm(i): return LABEL.get(i, i.replace("_", " "))
def num(x):
    s = f"{int(x):,}".replace(",", "{,}")
    return s

# ── Table 1: training corpus ──────────────────────────────────────────────
rows, n_train, n_hold = [], 0, 0
for t in man["training"]:
    mark = "$^a$" if t.get("hbvi_holdout") else ""
    if t.get("hbvi_holdout"): n_hold += 1
    else: n_train += 1
    rows.append((t["date_bce"], nm(t["id"]) + mark, REG[t["register"]],
                 t["date_bce"], t["date_sigma"], words.get(t["id"], 0), t["anchor"]))
hold_rows = []
for t in man["holdouts"]:
    hold_rows.append((t["date_bce"], nm(t["id"]) + "$^b$", REG[t["register"]],
                      t["date_bce"], t["date_sigma"], words.get(t["id"], 0), t["anchor"]))

rows.sort(key=lambda r: -r[0]); hold_rows.sort(key=lambda r: -r[0])
tot = sum(r[5] for r in rows + hold_rows)

L = []
L.append(r"\begin{table}[!ht]")
L.append(r"\centering")
L.append(r"\caption{\textbf{Dated corpus: %d training units and %d held-out units.}"
         % (len(rows), len(hold_rows)))
L.append(r"Date (BCE) is the scholarly consensus midpoint used as the training label;")
L.append(r"$\sigma$ is the accompanying uncertainty.  Sigmas were widened by")
L.append(r"approximately a factor of two, relative to earlier versions of this corpus,")
L.append(r"on every text whose date rests on literary-critical rather than")
L.append(r"synchronistic grounds.  Register: S = SBH (Standard Biblical Hebrew,")
L.append(r"equivalent to CBH), T = Transitional, L = LBH.  Word counts are token")
L.append(r"counts from the BHSA extraction used for all feature rates.")
L.append(r"$^a$Held out of the HB-VI model only.")
L.append(r"$^b$Held out of both models, and excluded from all feature screening.")
L.append(r"Whole-book Jeremiah is excluded from training entirely, because it")
L.append(r"contains the Jeremiah oracle holdout.}")
L.append(r"\begin{tabular}{llrrr}")
L.append(r"\hline")
L.append(r"\textbf{Unit} & \textbf{Register} & \textbf{Date (BCE)} &"
         r" \textbf{$\sigma$ (yr)} & \textbf{Words} \\")
L.append(r"\hline")
L.append(r"\multicolumn{5}{l}{\textit{Training}} \\")
for _, name, reg, d, s, w, _a in rows:
    L.append(f"{name:<22}& {reg} & {d:.0f} & {s:.0f} & {num(w)} " + r"\\")
L.append(r"\hline")
L.append(r"\multicolumn{5}{l}{\textit{Held out of both models}} \\")
for _, name, reg, d, s, w, _a in hold_rows:
    L.append(f"{name:<22}& {reg} & {d:.0f} & {s:.0f} & {num(w)} " + r"\\")
L.append(r"\hline")
L.append(r"\multicolumn{4}{l}{\textbf{Total}} & " + num(tot) + r" \\")
L.append(r"\hline")
L.append(r"\end{tabular}")
L.append(r"\label{tab:corpus}")
L.append(r"\end{table}")
open(DH.tab("tab_corpus.tex"), "w").write("\n".join(L) + "\n")

# ── Table: target date ranges ─────────────────────────────────────────────
g = pd.read_csv(DH.f("targets_generative.csv")).set_index("id")
r = pd.read_csv(DH.f("targets_ridge.csv")).set_index("id")
order = ["Song_Sea", "Song_Deborah", "D_Song", "Gen_JE", "Exo_JE", "Num_JE",
         "JE_source", "Lev_Holiness", "Lev_Priestly", "P_source",
         "D_Code", "D_Frame", "D_source", "Jer_DTR",
         "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy"]
order = [o for o in order if o in g.index]

T = []
T.append(r"\begin{table}[!ht]")
T.append(r"\centering")
T.append(r"\caption{\textbf{Date ranges for undated units.}  Intervals are")
T.append(r"conformal prediction intervals calibrated on leave-one-out residuals")
T.append(r"from the dated corpus, and are therefore distribution-free with")
T.append(r"guaranteed finite-sample marginal coverage; they replace the parametric")
T.append(r"intervals used previously, which achieved 52\% empirical coverage at a")
T.append(r"nominal 68\%.  Ranges are quoted earliest--latest in BCE.  $P_{\mathrm{post}}$")
T.append(r"is the posterior predictive probability that the unit postdates 586~BCE,")
T.append(r"reported as the \emph{lower} of the two model families at each row, so")
T.append(r"the column is conservative with respect to model choice.")
T.append(r"No unit has a 68\% interval confined to a single conventional period;")
T.append(r"the ranges, not the period labels, are the result.  D source and")
T.append(r"Deuteronomy are the same text and are not independent.")
T.append(r"$^\dagger$Interval extends past 1~BCE; the later bound is quoted in CE.}")
T.append(r"\begin{tabular}{lrrlrlr}")
T.append(r"\hline")
T.append(r" & & \multicolumn{2}{c}{\textbf{Generative}} &"
         r" \multicolumn{2}{c}{\textbf{Ridge}} & \\")
T.append(r"\cline{3-4}\cline{5-6}")
T.append(r"\textbf{Unit} & \textbf{Words} & \textbf{MAP} & \textbf{68\% range} &"
         r" \textbf{MAP} & \textbf{68\% range} & \textbf{$P_{\mathrm{post}}$} \\")
T.append(r"\hline")

def rng(hi, lo):
    """Format an earliest--latest range; render a negative BCE bound as CE."""
    if lo < 0:
        return f"{hi:.0f}--{abs(lo):.0f}~CE$^\\dagger$"
    return f"{hi:.0f}--{lo:.0f}"

for i in order:
    gg, rr = g.loc[i], r.loc[i]
    pp = min(gg["p_post_exilic"], rr["p_post_exilic"])
    T.append(f"{nm(i):<20}& {num(gg['n_words'])} & {gg['point']:.0f} & "
             f"{rng(gg['hi68'], gg['lo68'])} & {rr['point']:.0f} & "
             f"{rng(rr['hi68'], rr['lo68'])} & {pp:.2f} " + r"\\")
T.append(r"\hline")
T.append(r"\end{tabular}")
T.append(r"\label{tab:targets}")
T.append(r"\end{table}")
open(DH.tab("tab_targets.tex"), "w").write("\n".join(T) + "\n")

print("=== Table 1 corpus summary ===")
print(f"  training units       : {len(rows)}  (of which HB-VI holdouts: {n_hold})")
print(f"  dual holdouts        : {len(hold_rows)}")
print(f"  total dated tokens   : {tot:,}")
print(f"  date span            : {max(x[0] for x in rows+hold_rows):.0f}"
      f"--{min(x[0] for x in rows+hold_rows):.0f} BCE")
print()
print("=== drift vs the current main.tex Table 1 ===")
OLD = {"Isaiah_3": (450, 30), "Obadiah": (580, 30), "Joel": (400, 50),
       "Jonah": (400, 50), "Ezra": (350, 30), "Nehemiah": (350, 30),
       "Zechariah_2": (350, 50), "Esther": (300, 50), "Chronicles": (330, 30),
       "Hosea": (745, 20), "Micah": (725, 20), "Nahum": (650, 20),
       "Malachi": (460, 20)}
cur = {t["id"]: (t["date_bce"], t["date_sigma"]) for t in man["training"] + man["holdouts"]}
for k, (od, os_) in OLD.items():
    nd, ns = cur[k]
    if (od, os_) != (nd, ns):
        print(f"  {nm(k):<18} manuscript {od:.0f}+/-{os_:<3.0f} ->  manifest {nd:.0f}+/-{ns:.0f}")
missing = [nm(t['id']) for t in man['training'] if t['id'] in ('Lamentations','Ecclesiastes')]
print(f"  absent from manuscript table entirely: {', '.join(missing)}")
print("\nwrote tab_corpus.tex, tab_targets.tex")
