# Pre-submission checklist — PLOS ONE

State as of 2026-08-06, commit `6a09e8b`. Ordered by what would sink the
paper, not by effort.

---

## A. Blockers — these are wrong right now

**A1. Four orphaned supplementary figure captions.**
`S8 Fig`, `S10 Fig`, `S12 Fig`, `S13 Fig` still have `\paragraph*` caption
stubs, but their figure environments were pruned. They will render as
captions with no figure. S12's text still describes "HB-VI MAP estimates"
from the leaky run.
→ Delete the four stubs, or restore the figures if any is worth keeping.

**A2. The HB-VI section describes a model that reports nothing.**
Lines 795–892, roughly 98 lines: motivation, hierarchical architecture,
variational training, date posteriors, best-group selection. The results
subsection is now a deferral stub. A reviewer will ask why a model is
specified in full and never used.
→ Either re-run HB-VI under the corrected design (agnostic prior, in-fold
screening) and report it as a third family, or cut the section to a
paragraph in Limitations explaining why it was withdrawn. I lean toward
cutting: two families already carry the argument, and a third adds
multiplicity without adding evidence.

**A3. Stale v1 numbers survive in two places my sweep missed.**
My earlier sweep searched for specific strings (`402--406`, `16.8`, `63-text`)
and did not catch these:
- HB-VI motivation section: *"Haggai assigned to LBH: MAP = 361 BCE vs. 520
  BCE actual; error = 159 yr; holdout MAE for MLE-MVN = 134 yr."* The 134 yr
  figure is a v1 number, and the framing (HB-VI rescues MLE-MVN) is no longer
  supported by anything reported.
- Word n-gram subsection: *"returned MAP = 760 BCE — a 600-year [error]"*.
  The whole subsection describes an instrument whose results appear nowhere.
→ Do a full read-through hunting numbers, not a regex sweep. Every numeral in
the paper should trace to a file in `hebrew/results_v2/` or `greek/results/`.

**A4. Surviving figures were not regenerated under the corrected pipeline.**
I pruned figures tied to *deleted sections*. I did **not** verify that the
eight survivors reflect corpus v2 and the α=0.05 feature set. Suspect:
- `fig2_feature_correlations` — built on the 35-feature set, not 14
- `fig_s6_feature_loadings`, `fig_s7_covariance_heatmap` — same
- `fig_s9_elbo_convergence` — HB-VI diagnostic; dies with A2
- `fig8_prior_sensitivity` — pre-leakage-fix run
→ Check each one's provenance and regenerate or cut. Only `fig1_ordering` and
`fig2_timeline` are known-current.

**A5. S3 and S4 tables not regenerated.**
S3 (per-source driving features) and S4 (feature-exclusion robustness) still
reflect α=0.30 / 35 features. Both need the pipeline re-run at α=0.05.

---

## B. Scientific exposure — what a reviewer will attack

**B1. Five of 25 training labels are themselves linguistic judgements.**
This is the most serious item on the list, because the paper's own argument
makes it fatal if unaddressed.

| unit | date | stated anchor |
|---|---|---|
| Jonah | 400 ± 60 | "Aramaisms; late didactic narrative" |
| Ecclesiastes | 250 ± 120 | "Persian loanwords, Grecisms" |
| Esther | 300 ± 60 | "Persian loanwords" |
| Zechariah 9–14 | 350 ± 150 | "Greek-period allusions" |
| Joel | 400 ± 150 | "No king; Greeks named" |

Against 13 of 25 anchored to hard synchronisms (regnal dates, named kings,
Cyrus, Darius, the Maccabean crisis).

A paper that accuses the field of circularity, and then trains on labels
partly derived from linguistic dating, hands a reviewer the obvious riposte.
→ Run the whole analysis on the 13 synchronism-anchored units only, and
report it as the primary sensitivity check. If the ordinal result survives at
n=13, the paper is much harder to attack. If it doesn't, that needs saying.
Widened sigmas mitigate but do not remove the problem.

**B2. The headline metric was chosen after seeing the data.**
Pairwise ordering became the primary endpoint *because* period accuracy
failed. The permutation nulls are honest, but the choice of endpoint was not
pre-registered. A careful reviewer will call this a garden of forking paths.
→ Either state plainly that the endpoint was selected post hoc and treat the
Greek corpus as the confirmatory test (it was analysed after the endpoint was
fixed, which genuinely helps), or pre-register and re-run.

**B3. Nothing survives multiplicity correction at q = 0.05.**
Already stated in the paper. Make sure the abstract does not imply otherwise —
check the abstract's framing against the Validation section's own caveat.

**B4. Conformal exchangeability is assumed, not established.**
Calibration units are whole books; several targets are cross-book composites
(P draws on four books). The paper flags this. A reviewer may want a
composite-vs-book residual comparison. Cheap to run.

**B5. The genre correction rests on n = 5 narrative texts.**
The +112 yr adjustment that the post-exilic robustness argument leans on is
estimated from five texts, and is absent in the ridge family. Stated in the
paper, but it is thin. Consider a bootstrap CI on that estimate.

**B6. Non-independence in the target table.**
D source *is* Deuteronomy; sub-strata are nested in their composites. The
caption says so, but the count "19 undated units" implies more independent
evidence than exists. Consider reporting an effective count.

**B7. Reporting two model families.**
A reviewer could read this as hedging. The paper's defence — that
disagreement between families is itself informative, and that the genre bias
appears in one and not the other — should be stated explicitly as a design
choice, not left implicit.

---

## C. Judgment calls I made that need your sign-off

1. **Hornkohl moved from the exclusion list to suggested reviewers.** My
   reasoning: asking to exclude the field's most capable defender, in a paper
   whose central finding is a methodological failure in that tradition, reads
   as avoiding scrutiny. Your call, and reasonable people differ.
2. **The LBH-score ordering reversal** (D < JE < P became D < P < JE across
   corpus versions) is reported as evidence the metric is descriptive rather
   than probative. You may prefer a substantive interpretation.
3. **Δ_AB reported without absolute MAPs**, because the absolute values from
   that run contradict the conformal estimates by ~400 yr.
4. **Kings/Chronicles cut** at your direction. Nothing cites it.
5. **The self-correction framing.** The paper says "including in an earlier
   version of this work" in the abstract and expands in Validation. How much
   self-criticism belongs in the abstract versus a footnote is a real choice.
6. **Title.**

---

## D. PLOS ONE mechanics

- [ ] **Figure format.** PLOS requires TIFF or EPS, not PNG. 300–600 DPI,
      max 19.05 cm wide, under 10 MB. Files named `Fig1.tif`, `Fig2.tif`.
      Both new figures are currently PNG and will need conversion.
- [ ] **Supporting information naming.** `S1_Table.pdf`, `S1_Fig.tif` etc.,
      each cited in text as "S1 Table". Current numbering has gaps after the
      prunes — renumber sequentially.
- [ ] **Corresponding email inconsistency.** Manuscript says
      `adairaar@mit.edu`; cover letter says `adairaar@gmail.com`. Pick one.
- [ ] **Affiliation.** "Research Affiliate, MIT Physics" in the cover letter
      vs "Department of Physics, MIT" on the title page — confirm the form MIT
      wants for an unaffiliated/affiliate submission.
- [ ] **Data availability statement** — update for the restructured repo, and
      state explicitly what `archive/` is so a reviewer isn't confused by
      superseded material.
- [ ] **ORCID** linked to the submission.
- [ ] **Competing interests, funding, author contributions** — present but
      re-read after the rewrite.
- [ ] **Preprint policy** — decide on bioRxiv/arXiv/SocArXiv before or at
      submission.
- [ ] **Reference formatting** — 44 of 57 bib entries are now cited. Check the
      13 orphans are genuinely unneeded, and scan the rendered bibliography
      for the malformed entries bibtex warned about (BHSA2021, Kingma2014,
      Maclaurin2015, Perseus, TLG all have empty author/publisher fields).
- [ ] **Line numbers** — `lineno` is loaded. Confirm they render for review.
- [ ] **Reproducibility statement** — `build.sh` regenerates the manuscript;
      worth saying so explicitly in Data Availability.

---

## E. Worth doing, not strictly required

- **Repair or formally retire the resistant model.** Two documented defects
  (training-maximum overshoot instead of an exceedance probability; missing
  −mô, enclitic mem, yiqtol preterite, article and 'et rates). The Greek
  `archaism_resistance.py` result — morphology resists imitation R = +0.59,
  syntax does not, R = +0.01 — suggests the hypothesis is live even though
  this implementation isn't. Currently in `archive/`.
- **Pairwise ordering on the Torah targets.** Ordering P, D and JE relative to
  each other and to the dated corpus needs no absolute dates and may be the
  most defensible substantive claim available. Not yet run.
- **Bootstrap the conformal width** to show it is stable at n = 25.
- **A held-out corpus that did not exist when the method was built** — Ben
  Sira and Qumran non-biblical Hebrew are both in the repo's orbit and would
  be a genuinely blind test.

---

## Suggested order

1. A1–A5 (mechanical, half a day, unblocks everything)
2. B1 (the 13-unit sensitivity run — this is the one that decides whether the
   paper is defensible)
3. B2 decision, then C sign-offs
4. D mechanics last, once content is frozen
