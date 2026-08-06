# Undated units under the recommended model

Model: ~500-word chunks, 579 base features, inverse-density weighted (α = 1.0,
bandwidth 90), ridge with λ by leave-one-book-out, variance-matched.
Variance-match scale S = 1.83; LOBO MAE on the dated corpus 114.4 yr; conformal
half-widths ±138 yr (68%) and ±246 yr (90%), taken from the dated corpus's own
out-of-sample residuals.

## The archaic poems separate — and the model calls one pre-exilic

| unit | words | date | 68% interval | P(post-exilic) |
|---|---|---|---|---|
| **Song of Deborah** (Judg 5) | 462 | **814 BCE** | 952–676 | **0.04** |
| Song of the Sea (Exod 15) | 433 | 648 BCE | 786–510 | 0.32 |
| Song of Moses (Deut 32) | 782 | 529 BCE | 667–391 | 0.64 |

This is the first confident **pre-exilic** call the project has produced on any
undated unit. Under every previous design the model was one-sided and could
not place anything early.

The ordering is the scholarly one. Judges 5 is the text most widely held to be
the oldest surviving Hebrew poetry, Exodus 15 the second; the model puts them
first and second among nineteen targets and separates them from every prose
unit. It was trained only on prophetic and narrative books, with nothing
poetry-specific in the feature set and no poem in the training corpus.

**Caveat, stated plainly.** 814 BCE lies outside the training range of
760–167 BCE. Variance matching permits extrapolation beyond the observed range
— that is what makes the endpoints reachable — but an estimate past the last
anchor is an extrapolation and should be read as "earlier than anything in the
corpus," not as a date. Both songs are also single chunks, so they carry no
internal spread and rest entirely on the conformal interval.

## The documentary sources recover the classical sequence

| source | words | date | 68% interval | P(post-exilic) |
|---|---|---|---|---|
| JE | 37,752 | 420 BCE | 558–282 | 0.84 |
| D | 20,128 | 376 BCE | 514–238 | 0.96 |
| P | 54,435 | 330 BCE | 468–192 | 1.00 |

**JE → D → P, oldest to youngest.** That is the standard documentary
chronology. Bootstrapping over chunks, P(JE older than D older than P) =
**0.987**.

The earlier book-level analyses never produced a stable ordering of the three
sources; this one does, and it agrees with the sequence the field derived on
literary grounds a century and a half ago. The agreement is independent
evidence — nothing about the source partition or the training corpus encodes
that order.

All three remain post-exilic, now from a model demonstrated to be two-sided.

## Sub-strata and Pentateuch books

| unit | date | P(post-exilic) |
|---|---|---|
| Numbers JE | 442 | 0.84 |
| Genesis JE / D Code | 424 | 0.84 |
| Holiness Code | 396 | 0.92 |
| Exodus JE | 357 | 0.96 |
| D Frame | 322 | 1.00 |
| Leviticus P | 302 | 1.00 |
| Jeremiah Dtr | 447 | 0.84 |
| Genesis 413 · Numbers 352 · Exodus 347 · Leviticus 328 · Deuteronomy 376 | | 0.88–1.00 |

Note that D Frame (322) comes out later than D Code (424), reversing the
earlier analysis. Sub-strata are nested inside their composites and are not
independent evidence.

## How to read all of this

The 68% intervals are ±138 yr. Nothing here dates a text to a century. What
the model now supports, that it did not before:

1. A confident **pre-exilic** call is possible, and lands on the text the field
   independently regards as oldest.
2. The three documentary sources are **ordered** JE → D → P at 0.99 confidence.
3. All three are **post-exilic**, from a model that can and does say
   "pre-exilic" when the evidence warrants.

The first of those is what the earlier design could not do at all, and it is
the reason the third is now worth stating.
