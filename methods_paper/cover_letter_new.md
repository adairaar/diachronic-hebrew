# Cover Letter — PLOS ONE Submission

**Manuscript title:** The resolution of linguistic dating in Biblical Hebrew: calibrated ranges for the Pentateuchal sources, with cross-linguistic replication in Ancient Greek

**Corresponding author:** Aaron Adair (adairaar@gmail.com)
**Affiliation:** MIT Physics

---

Dear Editors,

I am submitting for your consideration a paper that measures the resolution of linguistic dating — how finely a text can actually be placed in time from its morphosyntax — and applies the answer to the composition of the Pentateuch.

Whether Biblical Hebrew morphosyntax carries a usable chronological signal has been disputed for two decades. One side treats linguistic dating as an established instrument; the other argues that its apparent successes are circular, because the models are validated against dates that are themselves partly linguistic judgements. The paper addresses this with a design in which no text contributes to its own prediction — feature screening, standardisation and model fitting all occur inside each cross-validation fold — and in which significance is established by permuting the date labels through the entire pipeline rather than by parametric assumption. Two model families are fitted and both are reported throughout, including where they disagree.

The positive result is that the signal is real. Across 25 dated units the models order text pairs chronologically at 69.8% and 67.8% accuracy against a permutation null near 45%, and leave-one-out rank correlations reach +0.58 and +0.49. The effect replicates in an unrelated language: a 63-text Ancient Greek corpus spanning 440 BCE–360 CE gives 60.3% and 58.9% pairwise ordering against a null near 40%.

The negative result is that this signal does not support absolute dating. Leave-one-out mean absolute error is 121–156 years in Hebrew against 137 years for the trivial predictor that assigns every text the corpus mean date; in Greek it is 205–295 years against 211 for the same trivial predictor. Four-way period assignment reaches 40–48%, and the exilic period — a 47-year window — is not resolvable at all.

I want to be explicit with the editors about one aspect of this submission. The analysis identifies a failure mode in holdout validation that I had previously fallen into myself: when a held-out text is dated under a prior centred on its own scholarly date, the model returns approximately that date regardless of the text's content, and does so most convincingly for the texts whose dates are best established. For the most securely anchored units in this corpus, the linguistic evidence contributes under 6% of posterior precision. An earlier version of this work reported holdout accuracies near 17 years on that basis; corrected, the same design yields roughly 104 years in Hebrew and 121 in Greek. The paper documents this in full, with the algebra and a per-text decomposition, because I think the failure mode is unlikely to be confined to my own work and because a reader is owed the correction alongside the claim.

What survives is narrower and, I think, more useful. Applied to the Pentateuchal sources under corrected conditions with distribution-free conformal intervals, the framework supports one substantive conclusion — that the Priestly, Deuteronomic and JE composites are post-exilic with probability at least 0.92 under both model families, and under a correction for the genre imbalance of the training corpus — while declining to place any of them in a particular post-exilic century. This finds independent corroboration in recent archaeological work (Adler 2022) and epigraphic analysis of Achaemenid-era sources (Barnea 2026). As a check on the ordinal signal, the Song of the Sea and the Song of Deborah emerge as the two earliest of nineteen undated units under both families, without having been used in training.

The manuscript is submitted exclusively to PLOS ONE and has not been published or submitted elsewhere. All code, extracted feature matrices, model outputs, and the scripts that regenerate every table and figure from the corpus manifest are available at https://github.com/adairaar/diachronic-hebrew.

I have no financial or personal competing interests to declare. The paper has one author; no co-author conflicts apply.

**Suggested reviewers.** The paper sits at the intersection of Bayesian statistics, corpus linguistics, and Hebrew Bible studies, and reviewers familiar with at least two of those areas would be best placed to evaluate it. Because the paper's central claims are now negative as well as positive, I would welcome reviewers on both sides of the underlying dispute.

- **Prof. Stephen Portnoy** (University of Illinois, Statistics Emeritus) — has published on statistical analysis of CBH/LBH feature distributions (Rachmuth, Portnoy & Wright, *Journal of Semitic Studies* 67.2, 2022) and would be well placed to evaluate the cross-validation and permutation methodology independently of the biblical studies content.

- **Dr. Cody Kingham** (University of Notre Dame, Data Science & Hebrew Bible) — has applied computational and machine-learning methods to the BHSA/ETCBC database, and is well positioned to evaluate both the corpus methodology and the statistical framework.

- **Prof. Wido van Peursen** (Vrije Universiteit Amsterdam) — a principal contributor to the BHSA/ETCBC corpus on which the feature extraction depends.

- **Dr. Martijn Naaijer** (University of Zurich) — has argued that existing linguistic dating methods rest on unsound statistical assumptions (Rezetko, Young, Ehrensvärd & Naaijer, *Bible and Interpretation*, 2025). Much of that critique is vindicated by the present analysis, and he is well placed to judge whether the corrections go far enough.

- **Prof. Aaron D. Hornkohl** (University of Cambridge) — has recently defended the methodology of linguistic dating (Hornkohl, *Bible and Interpretation*, 2026). Since this paper reports a failure mode affecting validation designs in that tradition, a reviewer disposed to defend the field's methods is exactly the right adversarial test, and I would rather the argument be examined by someone motivated to find its weaknesses than avoid that scrutiny.

- **Prof. Jacob L. Wright** (Emory University, Candler School of Theology) — whose recent work (*Why the Bible Began*, 2023) engages the compositional questions the results bear on.

Thank you for your time and consideration.

Sincerely,

Aaron Adair
Research Affiliate, MIT Physics
adairaar@gmail.com
