# Synthetic archaizing: how much apparent antiquity can a scribe buy?

Run: `archaize.py` (2026-08-06). Results in `archaize_results.csv`, log in `archaize.log`.
Feature-sensitivity companion in `sens.log`.

## The question

The oldest objection to any quantitative dating of Biblical Hebrew is that a late
author could deliberately imitate an earlier register. If archaizing is cheap, an
early estimate means nothing. The CBH/LBH literature debates this qualitatively.
It has never been priced.

It can be priced, because we can perform the archaizing ourselves and know the
ground truth exactly.

## Design

Take a securely dated **late** book. Replace its Late-Biblical-Hebrew forms with
their Classical counterparts at a controlled rate r, re-extract all 579 features
from the modified text, and re-date it with a model trained on the other 24 books
(unmodified). The substitutions are the classic diagnostics, applied LBH → CBH:

| LBH | CBH | gloss |
|---|---|---|
| שֶׁ־ (`C`) | אֲשֶׁר (`>CR`) | relative particle |
| אֲנִי (`>NJ`) | אָנֹכִי (`>NKJ`) | 1sg pronoun |
| מַלְכוּת (`MLKWT/`) | מַמְלָכָה (`MMLKH/`) | kingdom |
| אֵין (`>JN/`) | לֹא (`L>`) | negation (part of speech changed too) |

Substitution is at the token level via a proxy feature object, so every downstream
feature — lexeme rates, POS bigrams, phrase types, clause relations — sees the
altered text, exactly as it would if a scribe had written it that way.

r runs 0, 0.25, 0.50, 0.75, 1.00. r = 1 is *total* archaizing: not one LBH token
of these four types survives anywhere in the book.

## Result

```
unit            true     r=0     r=1   shift  swaps  per 1k w
Chronicles       330     275     275      +0     94       2.7
Esther           300     126     135      +9     42       9.1
Ecclesiastes     250     386     417     +31    142      33.5
Daniel           167     384     409     +25     47      13.7
Ezra             380     179     182      +3     13       2.5
Nehemiah         380     139     139      -0     28       3.6

mean apparent shift from 100% lexical archaizing: +11 yr
```

**Complete lexical archaizing buys a mean of 11 years, with a maximum of 31.**

The response is gradual, not thresholded — Ecclesiastes runs 386 → 392 → 396 →
418 → 417 across r, Daniel 384 → 394 → 399 → 399 → 409. There is no substitution
rate at which the estimate jumps.

### Read this with the density caveat

Three of the six books had almost nothing to swap: Ezra had 13 eligible tokens in
the entire book, Chronicles 94 across ~35,000 words. Their null results are
uninformative — you cannot archaize a text that already lacks the forms.

The informative cases are the three with real density:

| unit | swaps / 1k words | shift | yr per swap-per-1k |
|---|---|---|---|
| Ecclesiastes | 33.5 | +31 | 0.93 |
| Daniel | 13.7 | +25 | 1.82 |
| Esther | 9.1 | +9 | 0.99 |

So the exchange rate is roughly **1–2 years of apparent antiquity per lexical
substitution per 1000 words**. Ecclesiastes at r = 1 has one word in thirty
replaced — a density no real scribe sustains — and moves 31 years.

To buy 424 years at this rate a forger would need on the order of 250–450
substitutions per 1000 words. That is a quarter to nearly half of every word in
the text, and it exceeds the total corpus frequency of these lexemes: there are
not enough eligible tokens in Biblical Hebrew to spend.

## Why the model is hard to fool: where its leverage actually lives

`sens.log` computes years of apparent date per 1 SD shift in each feature.

Top levers (yr/SD):

```
lex_<MD[      -12.9    vs_hit        -7.4    lex_QRB/      +7.1
lex_RB/        -6.9    pb_advb_verb  +6.6    pb_prin_conj  -6.4
pb_intj_art    +6.4    lex_>NJ       -6.2    pb_prde_verb  -6.2
typ_CP         -6.0    pb_art_nmpr   +6.0    pb_prep_advb  +5.8
```

Leverage by feature family:

| family | Σ\|yr/SD\| | count | share |
|---|---|---|---|
| lexical | 603.3 | 250 | 47.0% |
| POS bigrams | 421.2 | 207 | 32.8% |
| verb morphology | 118.7 | 57 | 9.3% |
| phrase / clause | 90.9 | 40 | 7.1% |
| agreement / suffix | 35.7 | 20 | 2.8% |
| structural | 13.6 | 5 | 1.1% |

Lexical features carry 47% of the leverage — but spread across **250 features
none of which is worth more than 13 yr/SD**. No single word is a lever. The
remaining 53% sits in morphology and syntax: POS transition rates, verb stem and
aspect distributions, phrase-type and clause-relation profiles. These are the
features the sparsity analysis showed survive at 500-word chunk size precisely
because they are high-frequency (at 1000-word chunks `rate_yiqtol` ICC 0.79,
`rate_neg_lo` 0.57, `rate_qatal` 0.56, `rate_qal` 0.54 — all ρ ≈ 0.45–0.55 with
date, per `CHUNK_FINDINGS.md` §2), and they
are the ones a scribe cannot consciously monitor.

Formally: to displace an estimate by 424 years requires a coordinated shift of
length **6.43 SD** in the 579-dimensional feature space. Not a word swap — a
systematic re-profiling of the text's morphology and syntax.

## Consequence for the Song of the Sea

The Song of the Sea sits 424 years earlier than its Exodus prose frame. The
substitution budget above cannot produce that gap: full lexical archaizing of the
kind the CBH/LBH literature describes moves an estimate by tens of years, not
hundreds.

This does not prove the Song is early. It rules out one specific alternative —
that a late author produced the appearance of age by choosing archaic words. If
the Song's profile is manufactured, it was manufactured at the level of verbal
morphology and clause syntax, coherently, across 579 dimensions.

## Consequence for the method

The project originally set out to build an "archaizing-resistant" model by
selecting features a forger could not manipulate, and failed. The chunk-level
redesign produced that resistance as a by-product: forcing features to be dense
enough to be measurable in 500 words automatically demoted the rare lexical
shibboleths — which are exactly the manipulable features — and promoted
high-frequency morphology. Resistance was not designed in. It fell out of the
sample-size constraint.

## Limitations to state in the paper

1. Four substitution pairs is a small vocabulary of archaizing. A determined
   imitator would use more, and would also imitate syntax.
2. The test measures *lexical* archaizing only. It says nothing about whether
   morphosyntactic imitation is achievable, only that it would have to be the
   mechanism.
3. Three of six books lacked the density to be informative; the estimate rests
   substantially on Ecclesiastes and Daniel.
4. The r = 0 baselines carry the model's own genre errors (Ecclesiastes −136 yr,
   Daniel −217 yr, both too old). The *shift* is the quantity of interest and is
   unaffected, but the levels are not endorsements.

## Proposed follow-up (not yet run)

Greek supplies the labelled archaizing set Hebrew cannot: the Second Sophistic
Atticizers — Lucian, Aelian, Philostratus, Aelius Aristides — deliberately
imitated Classical Attic five to six centuries after the fact, with securely
known composition dates. An equivalent model trained on dated Greek prose and
applied to them measures how many centuries *skilled, sustained, deliberate*
archaizing actually buys — the number this Hebrew experiment can only bound from
below.
