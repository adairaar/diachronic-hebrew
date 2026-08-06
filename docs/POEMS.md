# The three poems, re-extracted at verse precision

## The bug

The project's unit specs defined the poems as whole chapters. They aren't:

| unit | poem proper | prose included | % prose |
|---|---|---|---|
| Song of the Sea | Exod 15:1–18, 228 w | 205 w (Miriam bridge, Marah narrative) | **47%** |
| Song of Moses | Deut 32:1–43, 576 w | 206 w (prose epilogue) | **26%** |
| Song of Deborah | Judg 5:2–31, 448 w | 14 w | 3% |

Narrative runs **+143 yr late** under this model, so prose contamination drags a
poem's estimate later. This flaw is inherited from the original book-level
pipeline, so every Song-of-the-Sea result this project has ever reported was
computed on a unit that is nearly half prose.

## Corrected estimates

| unit | words | date | 68% interval | P(post-exilic) |
|---|---|---|---|---|
| **Song of Deborah** (5:2–31) | 448 | **823 BCE** | 961–685 | **0.04** |
| **Song of the Sea** (15:1–18) | 228 | **797 BCE** | 935–659 | **0.04** |
| **Song of Moses** (32:1–43) | 576 | **610 BCE** | 747–472 | **0.56** |

Against the contaminated versions:

| unit | chapter (old) | poem only (new) | shift |
|---|---|---|---|
| Song of the Sea | 648, P=0.32 | **797, P=0.04** | **+149 yr earlier** |
| Song of Moses | 558, P=0.60 | 610, P=0.56 | +52 yr earlier |
| Song of Deborah | 814, P=0.04 | 823, P=0.04 | +9 yr |

The isolated prose confirms the mechanism: Exod 15:19–27 alone dates to 470 BCE
(P=0.72) and Deut 32:44–52 alone to 366 BCE (P=0.96).

## What this does to the Song of the Sea

It moves from "on the cusp, undetermined" to **confidently pre-exilic**,
alongside the Song of Deborah. The two poems the field independently regards as
the oldest surviving Hebrew verse are now the two earliest units in the study,
both at P(post-exilic) = 0.04.

## What this does to the Song of Moses

Less than it might. Deut 32:1–43 moves to 610 BCE but stays essentially at the
boundary, P(post-exilic) = 0.56 — undetermined rather than "probably
post-exilic." It remains the latest of the three by a wide margin.

Splitting it finds no internal structure: 32:1–25 gives 616 BCE and 32:26–43
gives 604 BCE. The divine-council material at 32:8–9 sits in the first block,
and that block is not linguistically earlier than the rest.

So the tension you identified is softened but not removed. On archaic theology
Deut 32 patterns with the oldest material; on the linguistic features measured
here it patterns roughly two centuries later than Exodus 15 and Judges 5. Those
are different kinds of evidence and they need not agree — archaizing diction is
exactly the mechanism by which they would come apart — but this model cannot
adjudicate that, since the resistant-model diagnostic built for the purpose
never functioned.

## The caveat that applies to all three

**Poetry is out of distribution.** Lamentations is the only poetry unit in the
25-book training corpus, and the model predicts it 149 yr too early (719 vs
570). If that bias generalises, all three poems are inflated: corrected by
−149 yr they would read Deborah 674, Sea 648, Moses 461, which would move the
Song of Moses firmly post-exilic and leave the other two late-pre-exilic.

One text is not enough to estimate a genre correction, so no correction is
applied. But the direction of the only available evidence is that these three
dates are too early, and that should be stated wherever they are reported.
