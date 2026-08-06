# Re-run status — corpus v2

> **Uncommitted work.** The sandbox git is blocked by stale lock files. Run this
> from your own terminal to capture everything from the 2026-08-05 session:
> ```
> cd "/Users/aaronadair/Documents/Claude/Projects/Diachronic Hebrew"
> rm -f .git/HEAD.lock .git/index.lock
> git add -A && git commit -m "Corpus v2 rebuild, diagnostics, Greek archaism test"
> git push origin main
> ```

## Resistant model rebuild — FIRST ATTEMPT FAILED

`hebrew/resistant_v3.py` runs but produces nothing usable. Two defects:

1. **Over-correction metric mis-specified.** Overshoot was measured as distance
   beyond the *training maximum*. The max is itself a noisy order statistic, so
   by construction almost nothing exceeds it — every text scored −2.0 to −2.6
   and zero features flagged. Fix: fit the per-feature training distribution and
   compute an exceedance probability, or use a fitted 95th percentile. The right
   question is "how improbable is this value under the dated-corpus
   distribution", not "how far past the most extreme observed text".

2. **−mô is not in the feature set.** `MORPH_RESIST` uses verb stem/form rates;
   `rate_prs` is the overall suffix rate, not the −mô form. The +41 SD
   observation that motivated the whole approach could not fire. Fix: add
   `F.prs.v(w) == "MW"`, enclitic mem, yiqtol preterite, article rate, 'et rate.

Also unresolved: morphological and syntactic resistant models disagree by up to
370 yr, and Song_Deborah pins at the 900 BCE grid edge (training ceiling).

The underlying hypothesis is still live — see `greek/archaism_resistance.py`,
where the Greek Atticizers show morphology resists imitation (R=+0.59) while
syntax does not (R=+0.01).

---


Audit and rebuild, August 2026. This file is the single source of truth for
what has been redone and what has not. Delete it once the re-run is complete.

## Decisions locked

| Question | Decision |
|---|---|
| New training texts | Obadiah, Joel, Zechariah 9–14, Lamentations, Ecclesiastes added |
| Date policy | Consensus midpoints, sigmas widened ~2x on literary-critically dated texts |
| Isaiah 56–66 | 400 ± 100 |
| Ezra / Nehemiah | 380 ± 60 |
| Haggai | Holdout for **both** models (carries the register-assignment argument) |
| Jer_oracle | Holdout for **both** models |
| Jeremiah whole-book | **Removed from training** (contained the Jer_oracle holdout) |
| Jer_DTR | Diagnostic target, never trained |

## Done

- `hebrew/corpus_manifest_v2.json` — 23 training units, 2 dual holdouts, 2 HB-VI-only holdouts
- `hebrew/build_feature_matrix_v2.py` — one-pass BHSA extraction, 44 units, reuses
  `extract_unit()` from `hierarchical_bayes/00_extract_features.py`
- `hebrew/data/feature_matrix_v2.csv` — 64 feature columns
- `hebrew/power_analysis_v2.py` → `results_v2/power_analysis.csv`
- `hebrew/loo_diagnostic_v2.py` → `results_v2/loo_diagnostic.csv`
- `hebrew/multiverse_v2.py` → `results_v2/multiverse.csv`
- `hebrew/screen_and_date_v2.py` → `results_v2/dating_v2.csv` (naive MVN, diagnostic only)

## Not done

The register-conditioned and hierarchical models have **not** been re-run
against `feature_matrix_v2.csv`. Everything below still reflects corpus v1.

1. `hebrew/03_hard_register_dating.py`
2. `hebrew/04_extended_analysis.py` (genre correction, archaism)
3. `hebrew/hierarchical_bayes/01_hb_vi_dating.py`
4. `hebrew/hierarchical_bayes/02_compare.py`
5. `hebrew/hierarchical_bayes/03_prior_sensitivity.py`
6. All 23 figures, all manuscript tables

## Blocking bugs to fix before re-running

### 1. HB-VI trains on its own holdout — `01_hb_vi_dating.py` line ~337

```python
jer_row["holdout"] = False      # puts Jer_oracle into TRAINING
```

The manuscript calls Jer_oracle "the primary held-out test" and states it was
"excluded from all feature selection and model training." It was not. The
reported HB-VI figure of 637 BCE is an in-sample fit. Set this to `True` and
re-run; expect the number to move and the validation claim to weaken.

### 2. Screening criterion is undefined across the codebase

| Script | p-threshold | LOO gate | LOO statistic |
|---|---|---|---|
| `11_comprehensive_dating.py` | 0.10 | 1.00 | sign consistency |
| `hebrew/03_hard_register_dating.py` | 0.25 | 0.30 | **significance retention** |
| `06_feature_mining.py` | 0.10 | 0.75 (advisory) | sign consistency |
| Manuscript text | 0.30 | 0.68 | sign consistency |

Four different specifications. Pick one and apply it everywhere.

### 3. The LOO filter has no rejection power at n = 23

`loo_diagnostic.csv`: under a permutation null, 100% of noise features that
pass p<alpha also clear the 68% LOO gate, at every alpha. FDP is identical
with and without the filter. Above |rho| ~ 0.2 sign-consistency is
deterministically 1.00, and p<0.30 at n=23 corresponds to |rho| > ~0.22, so
the gate sits where it can never fire.

Replace with k-fold (k=4) or register-block stability, or drop the LOO framing
and report Benjamini-Hochberg (q=0.10 retains 7 features).

### 4. Noise inflates confidence rather than cancelling

`multiverse.csv`: adding 6 and 12 pure-noise features to a clean p<0.01 model
shifts MAP by under ~50 yr (noise does cancel in location) but shrinks the 68%
HDI for **every** target (P_source 329 -> 285 -> 246). Chance-correlated
features get large fitted beta and small residual sigma, and the MVN likelihood
weights by 1/sigma. This is a mechanism for the documented 32%-coverage failure.

Implication: feature count must be justified on power grounds, not maximised.

## Power analysis result

n = 23, K = 55 candidates:

| alpha | retained | expected noise | est. FDP | min \|rho\| @80% power |
|---|---|---|---|---|
| 0.01 | 6 | 0.6 | 0.10 | 0.68 |
| 0.05 | 14 | 2.8 | 0.20 | 0.58 |
| 0.10 | 18 | 5.5 | 0.31 | 0.53 |
| 0.30 | 30 | 16.3 | **0.54** | 0.41 |

BH at q=0.10 independently retains 7 features. The defensible set is 6–14,
not 35. Recommended operating point: **alpha = 0.05**, with 0.01 and 0.10
reported as robustness bands.

Note: with J < N the residual covariance is no longer rank-deficient, so the
Tikhonov argument in the Methods section needs rewriting, not renumbering.

## Multiverse verdict (seven specifications, MAP range in BCE)

| Target | Range | Verdict |
|---|---|---|
| JE_source | 335–385 (50) | stable |
| P_source | 285–351 (66) | **stable — headline claim survives** |
| Jer_oracle | 525–609 (84) | stable |
| Habakkuk | 647–752 (105) | acceptable |
| Daniel | 271–405 (134) | biased old |
| Lev_Priestly | 231–367 (136) | weak |
| Jer_DTR | 199–373 (174) | weak |
| Song_Sea | 561–756 (195) | weak in absolute date — expected for an archaizing text |
| Haggai | 263–553 (290) | unstable under MLE-MVN |
| D_source | 100–491 (391) | **not identified** |
| D_Code | 173–591 (418) | **not identified** |

The archaism *diagnosis* is robust even where absolute dates are not: the sign
of `delta_arch` never flips across specifications. Song_Sea is archaizing in
all seven (+150 to +345), D_Code modernised in all seven (−181 to −599).

## Reporting recommendation

Report P and JE with the specification curve as primary evidence. State plainly
that D and D_Code are not identified at this corpus size, and why (no legal
prose in training; dominant late marker for D is the genre-confounded *asher*
rate). The Pentateuch-compilation argument in the Conclusion rests on P, so
dropping the D point estimate costs little.

## Also outstanding (from the pre-rebuild audit)

- Manuscript Table 1 describes a corpus that exists nowhere in the code
  (Obadiah/Joel/Zech 9–14 were listed but never extracted; word counts wrong).
  Rebuild it from `corpus_manifest_v2.json`.
- Table 4 "Mode A" column holds main-run HB-VI values while the Shift column
  is computed from the prior-sensitivity run; the two disagree by up to 115 yr
  because the main run uses best-group selection and the sensitivity run uses a
  fixed register. Diagnose after the HB-VI re-run, not before.
- P appears as 357 / 359 / 361 BCE in different sections; pick one convention.
