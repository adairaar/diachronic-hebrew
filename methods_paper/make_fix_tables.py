import pandas as pd, numpy as np, json, pathlib
B = "/mnt/user-data/uploads/Diachronic Hebrew"
P = pathlib.Path("/home/claude/paper")
m = pd.read_csv(f"{B}/hebrew/results_v2/master_results_v2.csv")
man = json.load(open(f"{B}/hebrew/corpus_manifest_v2.json"))
sig = {t["id"]: t["date_sigma"] for k in ("training", "holdouts") for t in man[k]}
hb = pd.read_csv(f"{B}/hebrew/hierarchical_bayes/results_v2/hb_vi_dating.csv")

NM = {"Isaiah_1":"Isaiah 1--39","Isaiah_2":"Isaiah 40--55","Isaiah_3":"Isaiah 56--66",
      "Zechariah_1":"Zechariah 1--8","Zechariah_2":"Zechariah 9--14","Daniel":"Daniel (Heb.)",
      "Jer_oracle":"Jeremiah oracle","Jer_DTR":"Jeremiah Dtr","Song_Sea":"Song of the Sea",
      "Song_Deborah":"Song of Deborah","D_source":"D source","P_source":"P source",
      "JE_source":"JE source","D_Code":"D Code","D_Frame":"D Frame","D_Song":"D Song",
      "Lev_Holiness":"Holiness Code","Lev_Priestly":"Leviticus P"}
nm = lambda i: NM.get(i, i.replace("_", " "))

# ── tab:leakage ──────────────────────────────────────────────────────────────
h = hb[hb["id"].isin(sig)].copy()
h["psd"] = h["id"].map(sig)
h["ratio"] = h["hb_post_std"] / h["psd"]
prec = 1/h["hb_post_std"]**2 - 1/h["psd"]**2
h["likc"] = np.where(prec > 0, 1/np.sqrt(np.where(prec > 0, prec, 1)), np.nan)
h["w"] = (1/h["likc"]**2) / (1/h["likc"]**2 + 1/h["psd"]**2) * 100
sel = ["Haggai","Zechariah_1","Daniel","Amos","Habakkuk","Ezekiel","Jer_oracle",
       "Chronicles","Ecclesiastes","Isaiah_3","Joel","Zechariah_2"]
h = h[h["id"].isin(sel)].sort_values("psd")

L = [r"\begin{table}[!ht]", r"\centering",
     r"\caption{\textbf{Decomposition of posterior precision for dated units.}",
     r"$\sigma_{\text{prior}}$ is the stated uncertainty on the scholarly date;",
     r"$\sigma_{\text{post}}$ is the posterior standard deviation returned when that",
     r"date is used as the prior mean.  The implied likelihood width is recovered as",
     r"$\sigma_L = (\sigma_{\text{post}}^{-2} - \sigma_{\text{prior}}^{-2})^{-1/2}$,",
     r"and the final column gives the share of posterior precision contributed by the",
     r"linguistic data.  The near-exact recoveries reported for Haggai,",
     r"Zechariah~1--8 and Daniel in earlier versions of this work correspond to data",
     r"shares below 6\%.  Median over all 25 dated units:",
     r"$\sigma_{\text{post}}/\sigma_{\text{prior}} = 0.851$, median",
     r"$|{\rm MAP} - d_u| = 10.4$~yr.}",
     r"\begin{tabular}{lrrrrr}", r"\hline",
     r"\textbf{Unit} & $\mathbf{d_u}$ & $\mathbf{\sigma_{\text{prior}}}$ &"
     r" $\mathbf{\sigma_{\text{post}}}$ & $\mathbf{\sigma_L}$ & \textbf{data share} \\",
     r" & (BCE) & (yr) & (yr) & (yr) & (\%) \\", r"\hline"]
for _, r in h.iterrows():
    L.append(f"{nm(r['id']):<20}& {r['scholarly_date_bce']:.0f} & {r['psd']:.0f} & "
             f"{r['hb_post_std']:.1f} & {r['likc']:.0f} & {r['w']:.1f} " + r"\\")
L += [r"\hline", r"\end{tabular}", r"\label{tab:leakage}", r"\end{table}"]
(P / "tab_leakage.tex").write_text("\n".join(L) + "\n")

# ── tab:subsources — Delta_AB only ───────────────────────────────────────────
p = m[m.ps_shift.notna()].copy().sort_values("ps_shift", key=abs)
T = [r"\begin{table}[!ht]", r"\centering",
     r"\caption{\textbf{Prior sensitivity: displacement between prior modes.}",
     r"$\Delta_{AB} = \text{MAP}(A) - \text{MAP}(B)$ measures how far a unit's",
     r"estimate moves when the scholarly prior (Mode~A) is replaced by the agnostic",
     r"prior (Mode~B).  Only the displacement is reported: the absolute MAP values",
     r"from this run are superseded by the leakage-free estimates of",
     r"Table~\ref{tab:targets}, but $\Delta_{AB}$ is a difference and is unaffected",
     r"by the level offset between the two analyses.  Classification follows the",
     r"thresholds defined above: data-driven ($|\Delta_{AB}| < 30$~yr), mildly",
     r"prior-influenced (30--80~yr), prior-dominated ($>80$~yr).}",
     r"\begin{tabular}{llrl}", r"\hline",
     r"\textbf{Unit} & \textbf{Role} & $\mathbf{\Delta_{AB}}$ \textbf{(yr)} &"
     r" \textbf{Classification} \\", r"\hline"]
for _, r in p.iterrows():
    role = {"holdout":"holdout","hbvi_holdout":"holdout (\\hbvi{})","target":"target"}.get(r["role"], r["role"])
    T.append(f"{nm(r['unit']):<20}& {role:<18}& {r['ps_shift']:+.0f} & {r['ps_verdict']} " + r"\\")
T += [r"\hline", r"\end{tabular}", r"\label{tab:subsources}", r"\end{table}"]
(P / "tab_subsources.tex").write_text("\n".join(T) + "\n")

v = m.set_index("unit")["lbh_score"]
print("wrote tab_leakage.tex, tab_subsources.tex")
print(f"  leakage rows {len(h)}, subsource rows {len(p)}")
print(f"  verdicts: {dict(p.ps_verdict.value_counts())}")
print(f"  LBH v2: D {v['D_source']:+.3f}  P {v['P_source']:+.3f}  JE {v['JE_source']:+.3f}"
      f"  Song_Sea {v['Song_Sea']:+.3f}  D_Code {v['D_Code']:+.3f}")
