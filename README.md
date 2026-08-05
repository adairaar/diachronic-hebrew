# Diachronic Dating of Ancient Texts: A Multivariate Bayesian Framework

Code and data for:

> Aaron Adair, "A multivariate Bayesian framework for diachronic dating of ancient texts: Validation on Biblical Hebrew and Ancient Greek." *PLOS ONE* (submitted 2026).

---

## Overview

This repository implements a fully probabilistic framework for inferring the composition date of ancient texts from their internal morphosyntactic features. Two complementary Bayesian models are provided:

- **MLE-MVN** — per-feature OLS regression with Tikhonov-regularized multivariate normal likelihood, evaluated on a 500-point BCE date grid.
- **HB-VI** — hierarchical Bayesian regression fitted by mean-field variational inference (MFVI) using the ADAM optimizer, providing group-level shrinkage and better-calibrated credible intervals.

Both models are validated on Biblical Hebrew (22 training texts, ~750 BCE–167 BCE) and Ancient Greek (63 training texts, ~450 BCE–250 CE). A resistant-model archaism diagnostic, genre-correction framework (Strategies B+D), and word n-gram instrument are also included.

---

## Repository structure

```
.
├── hebrew/                        Hebrew pipeline
│   ├── 00_corpus_manifest.py      Define training corpus and dates
│   ├── 01_feature_matrix.py       Merge ETCBC feature CSVs → feature_matrix.csv
│   ├── 02_register_classifier.py  Soft register assignment (SBH/Trans/LBH)
│   ├── 03_hard_register_dating.py MLE-MVN register-conditioned dating
│   ├── 04_extended_analysis.py    Genre correction and archaism diagnostics
│   ├── 05_subsource_dating.py     Date D/P/JE documentary sub-sources
│   ├── corpus_manifest.json       Corpus metadata (dates, registers, holdouts)
│   ├── data/
│   │   └── feature_matrix.csv     35 morphosyntactic features, 26 texts
│   ├── results/                   Output CSVs from scripts 02–05
│   └── hierarchical_bayes/
│       ├── 00_extract_features.py Extract features for sub-source units from BHSA
│       ├── 01_hb_vi_dating.py     HB-VI model (MFVI, ADAM, autograd)
│       ├── 02_compare.py          HB-VI vs MLE-MVN comparison
│       ├── 03_prior_sensitivity.py Mode A / Mode B / Mode C sensitivity analysis
│       └── results/               HB-VI output CSVs and prior sensitivity table
│
├── greek/                         Ancient Greek pipeline
│   ├── 00_corpus_manifest.py      Corpus definition (63 texts, 5 holdouts)
│   ├── 01_download_corpus.py      Download texts from Perseus Digital Library
│   ├── 02_preprocess.py           Tokenize, strip diacritics, build processed/
│   ├── 03_feature_extraction.py   Extract morphosyntactic + particle features
│   ├── 04_feature_screening.py    Spearman screening + LOO filter
│   ├── 05_mvn_dating.py           MLE-MVN dating for Greek corpus
│   ├── 06_holdout_validation.py   Five-holdout cross-validation
│   ├── 07_visualizations.py       Publication figures
│   ├── 08_register_classifier.py  LXX / Atticizing register classifier
│   ├── 09_calibrated_dating.py    Calibrated posteriors
│   ├── 10_hard_register_dating.py Register-conditioned MLE-MVN (mirrors Hebrew 03)
│   ├── corpus_manifest.json       Corpus metadata
│   ├── data/features/             Extracted feature matrices
│   ├── results/                   Dating results, holdout validation CSVs
│   └── hierarchical_bayes/
│       ├── 01_hb_vi_dating.py     HB-VI for Greek
│       ├── 02_compare.py          HB-VI vs MLE-MVN comparison
│       └── results/
│
├── methods_paper/                 Manuscript source
│   ├── main.tex                   PLOS ONE manuscript (LaTeX)
│   ├── main.pdf                   Compiled manuscript
│   └── figures/                   All 23 figures (300 DPI, PLOS ONE compliant)
│
├── 06_feature_mining.py           Full lexeme/morphology scan + LOO screening
├── 08_theoretical_features.py     ETCBC extraction: function words, verb forms
├── 10_morphosyntactic_dating.py   ETCBC extraction: Tier-1/2 morphosyntactic rates
├── 12_tier3_clause_features.py    ETCBC extraction: Tier-3 clause/phrase features
├── 17_word_ngram_dating.py        Word POS-tag n-gram dating instrument
├── 18_archaism_diagnostic.py      Full vs. resistant model archaism scatter
├── 19_torah_source_analysis.py    Chapter-level Torah source classification
│
├── master_dating_results.csv      Consolidated MLE-MVN results (all units)
├── feature_scan_full.csv          All ~550 screened feature candidates
├── feature_scan_robust.csv        LOO-robust subset used for dating
│
├── requirements.txt
└── .gitignore
```

---

## Requirements

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
```

The Hebrew pipeline additionally requires access to the **BHSA corpus** (Biblia Hebraica Stuttgartensia Amstelodamensis, version 2021) via the [Text-Fabric](https://github.com/annotation/text-fabric) framework:

```bash
pip install text-fabric
python -c "from tf.app import use; use('ETCBC/bhsa', version='2021')"
```

This downloads ~200 MB of corpus data on first run. The BHSA is released under a Creative Commons Attribution 4.0 license.

---

## Running the Hebrew pipeline

### Step 1 — Feature extraction (requires BHSA/Text-Fabric)

The three root-level extraction scripts query the BHSA directly and write CSV files used by the `hebrew/` pipeline:

```bash
python 08_theoretical_features.py     # → theoretical_features_training.csv
python 10_morphosyntactic_dating.py   # → morpho_training_rates.csv
python 12_tier3_clause_features.py    # → tier3_training_rates.csv
python 19_torah_source_analysis.py    # → source_feature_profiles.csv
```

Pre-extracted output CSVs are included in `hebrew/data/` so this step can be skipped if you are not modifying the feature set.

### Step 2 — Build the feature matrix

```bash
cd hebrew
python 01_feature_matrix.py           # → data/feature_matrix.csv
```

### Step 3 — MLE-MVN dating

```bash
python 02_register_classifier.py      # → results/register_probs.csv
python 03_hard_register_dating.py     # → results/hard_register_dating_hebrew.csv
python 04_extended_analysis.py        # → results/extended_analysis.csv (genre correction)
python 05_subsource_dating.py         # → results/subsource_dating.csv
```

### Step 4 — HB-VI dating

```bash
cd hierarchical_bayes
python 00_extract_features.py         # → results/extracted_features.csv (requires BHSA)
python 01_hb_vi_dating.py             # → results/hb_vi_dating.csv  (~5 min on laptop)
python 02_compare.py                  # → results/comparison_table.csv
python 03_prior_sensitivity.py        # → results/prior_sensitivity.csv
```

---

## Running the Ancient Greek pipeline

```bash
cd greek
python 00_corpus_manifest.py          # define corpus
python 01_download_corpus.py          # fetch texts from Perseus (~10 min, network required)
python 02_preprocess.py               # tokenize and strip polytonic diacritics
python 03_feature_extraction.py       # → data/features/feature_matrix.csv
python 04_feature_screening.py        # Spearman + LOO filter
python 05_mvn_dating.py               # MLE-MVN dating
python 06_holdout_validation.py       # five-holdout cross-validation
python 08_register_classifier.py      # LXX/Atticizing register classifier
python 10_hard_register_dating.py     # register-conditioned MLE-MVN
cd hierarchical_bayes
python 01_hb_vi_dating.py             # HB-VI for Greek
python 02_compare.py
```

**Note on polytonic Unicode:** All Greek token matching uses `strip_diacritics()` before regex comparison. Academic editions use the polytonic Unicode block (U+1F00–U+1FFF); matching without stripping silently produces zero counts. See `greek/03_feature_extraction.py` for details.

---

## Data sources

| Corpus | Source | License |
|--------|--------|---------|
| Biblical Hebrew | [BHSA v2021](https://github.com/ETCBC/bhsa) | CC BY 4.0 |
| Ancient Greek | [Perseus Digital Library](http://www.perseus.tufts.edu) | CC BY-SA 3.0 |
| Ancient Greek (supplementary) | [Thesaurus Linguae Graecae](http://stephanus.tlg.uci.edu) | Subscription (not redistributed) |

TLG texts are not included in this repository. The `greek/data/processed/` directory contains only Perseus-sourced texts; TLG texts must be obtained independently and placed there with matching filenames before running `greek/03_feature_extraction.py`.

---

## Key results

### Hebrew

| Text | MLE-MVN MAP | HB-VI MAP | Prior-sensitivity class |
|------|-------------|-----------|-------------------------|
| Oracle Jeremiah (holdout) | 562 BCE | 637 BCE | Data-driven |
| Song of the Sea (Exod 15) | 852 BCE (full) / 460 BCE (resistant) | 768 BCE | Data-driven (archaizing) |
| P source | 361 BCE | 404 BCE | Data-driven (\|Δ<sub>AB</sub>\| = 4 yr) |
| D source | 292 BCE | 394 BCE | Prior-dominated (\|Δ<sub>AB</sub>\| = 178 yr) |
| JE source | 435 BCE | 531 BCE | Prior-dominated (\|Δ<sub>AB</sub>\| = 172 yr) |
| D Frame | 274 BCE | 716 BCE | Data-driven |
| Haggai (holdout) | 361 BCE ❌ | 522 BCE ✓ | Prior-dominated |

MLE-MVN values are posterior modes under the agnostic prior (`master_dating_results.csv`, `map_full`); HB-VI values are from the main variational run (`hb_vi_dating.csv`, `hb_map_bce`). The two models differ by 40–100 yr on the source composites because HB-VI pools across register groups and propagates coefficient uncertainty. Both place all three Torah sources in the Persian-to-Hellenistic period.

HB-VI holdout MAE: **16.8 yr** vs. MLE-MVN: **134.2 yr** (improvement driven by soft register assignment).

The P source result is the most robustly data-driven finding in the paper: a prior sweep from a flat prior through a Mosaic-authorship prior *N*(1200, 100²) returns a Persian-period MAP under every scenario (see S6 Table).

### Ancient Greek (cross-language validation)

All five holdouts are recovered within ~30 yr of their independently established dates by the register-conditioned model (`greek/results/hard_register_dating.csv`):

| Holdout | Established date | MAP | Error |
|---------|------------------|-----|-------|
| Polybius, *Histories* | 160 BCE | 148 BCE | 12 yr |
| Mark | 70 CE | 69 CE | 1 yr |
| Matthew | 85 CE | 80 CE | 5 yr |
| Luke | 120 CE | 120 CE | <1 yr |
| Diogenes Laërtius | 230 CE | 256 CE | 26 yr |

Note that the *unconditioned* likelihood (no register assignment, `holdout_validation.csv`, `lik_only_map_ce`) is 80–345 yr off for these same texts. Register conditioning is what makes the temporal signal recoverable — the raw feature likelihood alone is not sufficient.

---

## Citation

If you use this code or data, please cite:

```
Aaron Adair (2026). A multivariate Bayesian framework for diachronic dating
of ancient texts: Validation on Biblical Hebrew and Ancient Greek.
PLOS ONE. https://github.com/adairaar/diachronic-hebrew
```

---

## License

Code: MIT License. See `LICENSE` for details.

Data files (`hebrew/data/`, `hebrew/results/`, `greek/data/features/`, `greek/results/`): CC BY 4.0, consistent with the upstream BHSA and Perseus licenses.
