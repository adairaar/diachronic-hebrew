# main.tex salvage map + honest target dating

2026-08-06. PLOS ONE format retained throughout (`plos2025.bst`, template
intact). Nothing here requires reformatting.

---

## Part 1 — Torah source dating, re-derived blind

Model fit on all 25 dated texts; targets predicted with an agnostic prior and
never contributing to their own fit. Intervals are **conformal**, calibrated on
leave-one-out residuals — distribution-free, with guaranteed finite-sample
marginal coverage. This replaces the parametric intervals that achieved 52%
coverage at a nominal 68%.

Conformal half-widths: generative ±233 yr (68%), ridge ±174 yr (68%).

### P(post-exilic) — the claim that holds

| source | generative | ridge |
|---|---|---|
| D_Frame | 1.00 | 1.00 |
| D_Code | 0.96 | 1.00 |
| D_source | 0.96 | 1.00 |
| P_source | 0.96 | 0.92 |
| Lev_Priestly | 0.96 | 0.96 |
| JE_source | 0.92 | 0.92 |
| Lev_Holiness | 0.76 | 0.92 |

**All three documentary sources are post-exilic with probability ≥ 0.92 under
both model families.** This survives the leakage correction. It does not depend
on absolute dates, only on the position of a mass of predictive probability
relative to a single boundary (586 BCE) — which is exactly the kind of claim
the data supports.

### What does NOT hold: which post-exilic period

Not one of the 19 targets has a 68% interval confined to a single period. All
span 2–4 periods. Cross-family agreement on the single most likely period is
**47% (9/19)**, median point difference 59 yr. Generative pushes targets
Hellenistic; ridge says Persian for most.

So: "P is post-exilic" is supportable. "P is Persian" or "P is 361 BCE" is not.

### Internal validity check the model passes

Song of the Sea and Song of Deborah — the two texts most widely argued to be
the oldest Hebrew poetry — come out as the **two earliest of all 19 targets in
both families**, and are the only targets whose most likely period is
pre-exilic (P(post-exilic) 0.40–0.48). Neither was used in training. This is a
genuine out-of-sample check on an independently motivated hypothesis, and it is
worth reporting as such.

### Caveats to state in the paper

1. Conformal validity assumes exchangeability between calibration texts and
   targets. Calibration texts are whole books/sections; several targets are
   cross-book composites (P_source draws on four books). Flag explicitly.
2. |LOO residual| vs word count: rho = −0.23, p = 0.27 — no detectable size
   dependence, so a single conformal width is defensible. Say so.
3. Deuteronomy and D_source are the same text and must not be reported as
   independent results.

---

## Part 2 — What survives in main.tex (2405 lines, 9 tables, 18 figures, 57 refs)

### Survives intact — no change needed

| lines | section | note |
|---|---|---|
| 234–448 | **Background: diachronic Biblical Hebrew** | CBH/LBH distinction, diachronic vs dialectal debate, circularity problem, prior computational work. ~215 lines. The scholarship is untouched by any of this. |
| 452–463 | BHSA database | |
| 529–650 | Feature extraction, word n-gram model | Method description; extraction code is unaffected |
| 1514–1569 | **Scholarly context for late Torah dating** | Now *better* supported than before — Adler 2022 and Barnea 2026 line up with a post-exilic P/D/JE finding that no longer rests on a circular holdout |
| all | **references.bib — all 57 entries** | |

### Survives with edits

| lines | section | required edit |
|---|---|---|
| 464–528 | Training corpus / Table 1 | Rebuild Table 1 from `corpus_manifest_v2.json`; current table describes a corpus that exists nowhere in the code, with wrong word counts |
| 651–728 | MLE-MVN model | Math stands. The Tikhonov justification must be **rewritten, not renumbered**: with J < N the residual covariance is no longer rank-deficient, so the stated motivation no longer applies |
| 729–816 | HB-VI architecture | Architecture fine; §817 results die |
| 864–950 | **Prior sensitivity analysis** | Promote. This section diagnosed prior domination as a risk; it is now the central empirical finding. Expand rather than cut |
| 1122–1181 | Genre correction | Method stands, results need re-running |
| 1366–1513 | Limitations | Expand substantially |

### Dies

| lines | section | why |
|---|---|---|
| 951–1121 | **Archaism detection: resistant model** | `resistant_v3.csv` returns `n_feat_overshoot = 0` for all 44 units. The instrument does not fire. ~170 lines |
| 1185–1222 | Cross-language validation (Greek) | Circular — same prior-leakage bug, independently. Reported MAE 8.7 yr, honest 120.7 yr |
| 1223–1234 | Oracle Jeremiah primary holdout | Circular |
| 1235–1269 | LOO cross-validation and calibration | Replace with conformal |
| 1270–1284 | Kings/Chronicles parallel validation | Cut at author's direction, 2026-08-06 |
| 1285–1362 | HB-VI vs MLE-MVN head-to-head | Both sides leaky; comparison meaningless |
| 92–130, 1615–1653 | Abstract, Conclusion | Rewrite to the new claim |

### Rough accounting

Of 2405 lines, roughly **750 survive structurally** and the full bibliography
survives. About 350 lines are deleted outright; the rest is results prose
requiring re-running rather than rethinking.

---

## Part 3 — The paper this becomes

**Claim.** Hebrew morphosyntactic features order texts chronologically well
above chance (pairwise ordering 69.8% of 295 pairs, p = 4e-12; LOO Spearman
rho = +0.58 / +0.49 across two model families, p = 0.016 / 0.024) but do not
support absolute dating (LOO MAE 121–156 yr against 137 yr for predicting the
corpus mean) or reliable period assignment (4-period exact 40–48%).

**Application.** Applied to the Pentateuchal sources under leakage-free
conditions with calibrated conformal intervals, this supports one substantive
conclusion — P, D, and JE are post-exilic at P >= 0.92 — while explicitly
declining to date them further.

**Why this is publishable.** The field's live dispute (Hurvitz vs. Rezetko /
Young / Ehrensvard / Naaijer) is precisely about whether the linguistic signal
is real and whether the methods are circular. This answers both with a
pre-registered, permutation-tested, leakage-free design: the signal is real,
the circularity is real, and the two facts are compatible. The negative
methodological result and the positive substantive result reinforce each other
rather than competing.
