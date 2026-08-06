# The resolution of linguistic dating in Biblical Hebrew

Code and data for:

> Aaron Adair, "The resolution of linguistic dating in Biblical Hebrew:
> calibrated ranges for the Pentateuchal sources, with cross-linguistic
> replication in Ancient Greek." *PLOS ONE* (in preparation, 2026).

---

## What this repository contains

A leave-one-out framework for estimating how finely an ancient text can be
placed in time from its morphosyntax, applied to 25 dated units of the Hebrew
Bible and 63 dated Ancient Greek texts.

Every out-of-sample result uses an agnostic prior. Feature screening,
standardisation and model fitting are performed inside each cross-validation
fold. Significance is established by permuting the date labels and re-running
the entire pipeline. Two model families are fitted and both are reported.

### Headline results

| | Hebrew (n=25) | Greek (n=63) |
|---|---|---|
| pairwise ordering accuracy | 69.8% / 67.8% | 60.3% / 58.9% |
| permutation null | 47.6% / 43.6% | 39.2% / 40.4% |
| *p* | 0.022 / 0.029 | 0.037 / 0.050 |
| LOO Spearman rho | +0.575 / +0.494 | +0.363 / +0.265 |
| LOO MAE | 156.2 / 120.9 yr | 294.5 / 205.4 yr |
| **constant-predictor MAE** | **137.3 yr** | **210.9 yr** |

Values are generative / ridge. The corpus-mean row is the number every MAE in
this literature should be read against.

The signal is ordinal and real; absolute dating is not supported. Applied to
the Pentateuchal sources, the framework places P, D and JE after 586 BCE with
probability >= 0.92 under both families, and declines to locate them within a
particular post-exilic period.

---

## Layout

```
.
├── docs/                    findings and audit trail
│   ├── VERIFICATION_2026-08-06.md   prior-leakage analysis
│   ├── PERIOD_FINDINGS.md           LOO period evaluation
│   ├── SALVAGE_MAP.md               what survived the v1 audit
│   └── REBUILD_2026-08-06.md        manuscript rebuild log
│
├── hebrew/                  current Hebrew pipeline
│   ├── corpus_manifest_v2.json      25 dated units, dates and sigmas
│   ├── build_feature_matrix_v2.py   one-pass BHSA extraction
│   ├── data/feature_matrix_v2.csv   64 features, 44 units
│   ├── period_loo2.py               LOO + permutation, both families
│   ├── target_dating.py             conformal intervals for undated units
│   ├── make_tables.py               -> tab_corpus.tex, tab_targets.tex
│   ├── make_fig_ordering.py         -> Fig 1
│   ├── make_fig_timeline.py         -> Fig 2
│   ├── power_analysis_v2.py         screening power / FDP
│   ├── loo_diagnostic_v2.py         LOO-filter rejection power
│   ├── multiverse_v2.py             specification curve
│   ├── hierarchical_bayes/          HB-VI model and prior sensitivity
│   └── results_v2/                  all current output CSVs
│
├── greek/                   cross-linguistic replication
│   ├── 01_download_corpus.py        Perseus fetch
│   ├── 02_preprocess.py             tokenise, strip polytonic diacritics
│   ├── 03_feature_extraction.py     -> data/features/feature_matrix.csv
│   ├── greek_loo.py                 identical protocol to hebrew/period_loo2.py
│   └── results/greek_loo_results.csv
│
├── methods_paper/           manuscript and its build chain
│   ├── main.tex                     v1 source; splice.py reads this
│   ├── main_new.tex                 CURRENT DRAFT (generated)
│   ├── build.sh                     one command, full rebuild
│   ├── splice.py                    assemble main_new from main + fragments
│   ├── patch_intro.py               prose patches, passes 1-3
│   ├── patch_final.py               p-value corrections, Greek section
│   ├── insert_figs.py               figure insertion, float pruning
│   ├── make_fix_tables.py           tab_leakage, tab_subsources
│   ├── make_stables.py              S2, S5
│   ├── new_*.tex, tab_*.tex         section and table fragments
│   └── figures/                     8 figures, 300 DPI
│
└── archive/                 v1 material, superseded (280 files)
```

**`archive/` is superseded work retained for provenance.** Nothing in the
current draft depends on it. Delete the folder if you want a minimal tree; git
history retains everything.

---

## Rebuilding the manuscript

```bash
cd methods_paper && ./build.sh
```

Runs splice -> patch -> patch -> figure insertion -> pdflatex -> bibtex ->
pdflatex x3. Every patch fails loudly on a missing anchor rather than silently
skipping, so an edit to `main.tex` that breaks a splice point halts the build
instead of producing a quietly wrong PDF.

## Reproducing the analysis

```bash
pip install -r requirements.txt

# Hebrew feature extraction (requires BHSA via Text-Fabric, ~200 MB first run)
pip install text-fabric
python -c "from tf.app import use; use('ETCBC/bhsa', version='2021')"
python hebrew/build_feature_matrix_v2.py

# Hebrew analysis
python hebrew/period_loo2.py 0.05 2000     # LOO + permutation null
python hebrew/target_dating.py             # conformal intervals
python hebrew/make_fig_ordering.py hebrew/make_fig_timeline.py

# Greek
python greek/01_download_corpus.py && python greek/02_preprocess.py
python greek/03_feature_extraction.py
python greek/greek_loo.py 0.05 500
```

Pre-extracted feature matrices are committed, so the BHSA and Perseus steps can
be skipped unless you are changing the feature set.

---

## A note on the earlier version

An earlier version of this work reported holdout accuracies near 17 years for
Hebrew and "within ~30 yr" for Greek. Those figures were produced by a design in
which each held-out text was dated under a prior centred on its own scholarly
date. For the most securely anchored texts, the linguistic evidence contributed
under 6% of posterior precision. Corrected, the same design yields roughly 104
years in Hebrew and 121 in Greek.

`docs/VERIFICATION_2026-08-06.md` documents this with the algebra and a per-text
decomposition. The resistant-model archaism diagnostic and the genre-correction
procedure from that version have been withdrawn — the former returned zero
flagged features across all 44 units. Both remain in `archive/`.

---

## Data sources

| Corpus | Source | License |
|--------|--------|---------|
| Biblical Hebrew | [BHSA v2021](https://github.com/ETCBC/bhsa) | CC BY 4.0 |
| Ancient Greek | [Perseus Digital Library](http://www.perseus.tufts.edu) | CC BY-SA 3.0 |

Greek raw and processed texts are gitignored; `01_download_corpus.py` fetches
them.

## License

Code: MIT (see `LICENSE`). Data files: CC BY 4.0, consistent with upstream BHSA
and Perseus licenses.
