"""S-tables regenerated from corpus v2 outputs."""
import pandas as pd, json, pathlib
B="/mnt/user-data/uploads/Diachronic Hebrew"; P=pathlib.Path(".")
pw=pd.read_csv(f"{B}/hebrew/results_v2/power_analysis.csv")
mv=pd.read_csv(f"{B}/hebrew/results_v2/multiverse.csv")
fm=pd.read_csv(f"{B}/hebrew/data/feature_matrix_v2.csv")
man=json.load(open(f"{B}/hebrew/corpus_manifest_v2.json"))
NM={"Isaiah_1":"Isaiah 1--39","Isaiah_2":"Isaiah 40--55","Isaiah_3":"Isaiah 56--66",
    "Zechariah_1":"Zechariah 1--8","Zechariah_2":"Zechariah 9--14","Daniel":"Daniel (Heb.)",
    "Jer_oracle":"Jeremiah oracle","Jer_DTR":"Jeremiah Dtr","Song_Sea":"Song of the Sea",
    "Song_Deborah":"Song of Deborah","D_source":"D source","P_source":"P source",
    "JE_source":"JE source","D_Code":"D Code","D_Frame":"D Frame","D_Song":"D Song",
    "Lev_Holiness":"Holiness Code","Lev_Priestly":"Leviticus P","Gen_JE":"Genesis JE",
    "Exo_JE":"Exodus JE","Num_JE":"Numbers JE"}
nm=lambda i: NM.get(i,i.replace("_"," ")); num=lambda x:f"{int(x):,}".replace(",","{,}")

# S2: power analysis
L=[r"\begin{table}[!ht]",r"\centering",
   r"\caption{\textbf{S2 Table.\ Screening power and false-discovery cost at $N=23$.}",
   r"For each nominal $\alpha$: features retained from $K=55$ candidates, the number",
   r"expected under the null, the estimated false discovery proportion, and the minimum",
   r"$|\rho|$ detectable at 80\% power.  Benjamini-Hochberg at $q=0.10$ independently",
   r"retains 7 features.  The operating point used throughout is $\alpha=0.05$; the",
   r"35-feature specification of earlier versions of this work corresponds to",
   r"$\alpha \approx 0.3$, where more than half the retained features are expected to",
   r"be noise.}",
   r"\begin{tabular}{rrrrr}",r"\hline",
   r"$\bm{\alpha}$ & \textbf{retained} & \textbf{expected null} & \textbf{est.\ FDP} &"
   r" \textbf{min $|\rho|$ @80\%} \\",r"\hline"]
L=[x.replace(r"\bm",r"\mathbf") for x in L]
for _,r in pw.iterrows():
    L.append(f"{r['alpha']:.2f} & {int(r['retained'])} & {r['expected_null']:.1f} & "
             f"{r['est_FDP']:.2f} & {r['min_rho_80pct']:.2f} " + r"\\")
L+=[r"\hline",r"\end{tabular}",r"\label{tab:s2_power}",r"\end{table}"]
P.joinpath("tab_s2_power.tex").write_text("\n".join(L)+"\n")

# S5: word counts, all units
rows=[(t["id"],"training" if k=="training" else "holdout") for k in ("training","holdouts") for t in man[k]]
ids=[r[0] for r in rows]
tgt=[i for i in fm["id"] if i not in ids]
W=[r"\begin{table}[!ht]",r"\centering",
   r"\caption{\textbf{S5 Table.\ Token counts for all units.}",
   r"Counts are from the BHSA extraction used for every feature rate in this paper.",
   r"Source composites overlap the books that contain them and are not independent;",
   r"D source and Deuteronomy are the same text.}",
   r"\begin{tabular}{lrl}",r"\hline",
   r"\textbf{Unit} & \textbf{Words} & \textbf{Role} \\",r"\hline"]
wd=dict(zip(fm["id"],fm["n_words"]))
for i,role in sorted(rows,key=lambda r:-wd.get(r[0],0)):
    W.append(f"{nm(i)} & {num(wd.get(i,0))} & {role} " + r"\\")
W.append(r"\hline")
for i in sorted(tgt,key=lambda x:-wd.get(x,0)):
    W.append(f"{nm(i)} & {num(wd.get(i,0))} & target " + r"\\")
W+=[r"\hline",r"\end{tabular}",r"\label{tab:s5_words}",r"\end{table}"]
P.joinpath("tab_s5_words.tex").write_text("\n".join(W)+"\n")
print(f"wrote tab_s2_power.tex ({len(pw)} rows), tab_s5_words.tex ({len(rows)} dated + {len(tgt)} targets)")
print("multiverse columns:", list(mv.columns)[:8])
