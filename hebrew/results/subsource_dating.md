# Sub-source Dating Results

Sub-sources separated by chapter ranges within Deuteronomy and Leviticus.
N-gram dates are chapter-weighted aggregations from `torah_chapter_dates.csv`.
Morpho dates are from Script 13 (full morphosyntactic model, Script 11).

## D Sub-components

| Unit | Prior (BCE) | Word n-gram | Word CI68 | Morpho MAP | Morpho CI68 | mean_lbh | Class |
|------|------------|------------|---------|-----------|------------|----------|-------|
| D_source | 625 ± 100 | — | — | 292 | 257–329 | — | — |
| D_Code | 625 ± 100 | 400 | 358–442 | 292 | 232–350 | 0.279 | Archaic (CBH-like) |
| D_Frame | 600 ± 100 | 443 | 420–468 | 274 | 227–320 | 0.502 | Mixed/selective |
| D_Song | 900 ± 200 | 520 | 520–520 | 679 | 608–751 | 0.183 | Archaic (CBH-like) |

## P Sub-components

| Unit | Prior (BCE) | Word n-gram | Word CI68 | Morpho MAP | Morpho CI68 | mean_lbh | Class |
|------|------------|------------|---------|-----------|------------|----------|-------|
| P_source | 600 ± 100 | — | — | 361 | 329–393 | — | — |
| Lev_Holiness | 575 ± 75 | 498 | 463–537 | 301 | 234–366 | 0.425 | Mixed/selective |
| Lev_Priestly | 600 ± 100 | 435 | 394–485 | 465 | 414–513 | 0.526 | Mixed/selective |

## JE Sub-components

| Unit | Prior (BCE) | Word n-gram | Word CI68 | Morpho MAP | Morpho CI68 | mean_lbh | Class |
|------|------------|------------|---------|-----------|------------|----------|-------|
| JE_source | 800 ± 75 | — | — | 435 | 403–467 | — | — |
| Gen_JE | 800 ± 100 | 504 | 490–518 | 416 | 384–449 | 0.714 | Modern (LBH-like) |
| Exo_JE | 800 ± 100 | 480 | 454–506 | 465 | 421–509 | 0.588 | Modern (LBH-like) |
| Num_JE | 800 ± 100 | 616 | 586–645 | 479 | 416–541 | 0.730 | Modern (LBH-like) |

## Notes on Data Sources

- **N-gram dates** are word-count weighted averages of per-chapter char and word
  n-gram MAP estimates from `torah_chapter_dates.csv`, bootstrapped for 68% CI.
- **Morpho MAP / CI68** comes from Script 13 (`subsource_dating.csv` and
  `master_dating_results.csv`), which applied the full 36-feature morphosyntactic
  model to BHSA chapter-range text extractions.
- **mean_lbh** is the mean LBH score across all 36 features (from
  `subsource_archaism.csv`); higher = more LBH-like.
- **BHSA morpho extraction** for sub-sources requires text-fabric + BHSA data.
  The new pipeline (Scripts 00–04) currently only has morpho feature profiles
  for whole-source D, P, JE aggregates (`source_feature_profiles.csv`).
  To add sub-sources to Scripts 02–03, run Script 19 extended with sub-unit
  chapter-range definitions.
