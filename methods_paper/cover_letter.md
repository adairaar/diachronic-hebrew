# Cover Letter — PLOS ONE Submission

**Manuscript title:** A Multivariate Bayesian Framework for Diachronic Dating of Ancient Texts: Validation on Biblical Hebrew and Ancient Greek

**Corresponding author:** Aaron Adair (adairaar@gmail.com)
**Affiliation:** MIT Physics

---

Dear Editors,

I am submitting for your consideration a methods paper introducing a fully probabilistic framework for diachronic dating of ancient texts, validated on Biblical Hebrew and Ancient Greek corpora. The paper pairs two Bayesian models — a multivariate Gaussian (MLE-MVN) and a hierarchical variational inference model (HB-VI) — with an archaism diagnostic derived from the divergence between full-feature and syntactically conservative "resistant" models. An explicit genre-correction procedure addresses register confounds that have long complicated linguistic dating in this field.

The paper was designed from the outset to be transparent about what the model can and cannot do. A leave-one-out calibration analysis shows that the nominal 68% credible intervals achieve only 32% actual coverage, which the paper reports directly and corrects for rather than concealing. A prior-sensitivity analysis classifies each out-of-sample text as data-driven or prior-dominated, and a quantitative prior-sweep experiment demonstrates that the late dating of Priestly and Deuteronomic source composites cannot be recovered by prior assumption alone: even a Mosaic-authorship prior centered at 1200 BCE returns Persian-period MAP dates for all three Torah sources. These results find independent corroboration in recent archaeological work (Adler 2022) and epigraphic analysis of Achaemenid-era sources (Barnea 2026), and the Discussion explicitly situates them within the range of positions that have been advanced in the scholarly literature, including the mainstream critical consensus (Wellhausen, Baden) and more recent revisionist arguments.

Cross-language validation on sixty-three Ancient Greek texts, with five independent holdouts recovering all five established dates within approximately thirty years, suggests the framework is not specific to Hebrew and may be applicable to any ancient language corpus with independently anchored dates.

The manuscript is submitted exclusively to PLOS ONE and has not been published or submitted elsewhere. All code, extracted feature matrices, and model outputs are available in the public GitHub repository at https://github.com/adairaar/diachronic-hebrew. The corpus draws on the publicly available BHSA/Text-Fabric morphological database of the Hebrew Bible and the Perseus Digital Library for Ancient Greek.

I have no financial or personal competing interests to declare. The paper has one author; no co-author conflicts apply.

**Suggested reviewers.** The paper sits at the intersection of Bayesian statistics, corpus linguistics, and Hebrew Bible studies, and reviewers familiar with at least two of those areas would be best placed to evaluate it.

- **Prof. Stephen Portnoy** (University of Illinois, Statistics Emeritus) — has published on statistical analysis of CBH/LBH feature distributions (Rachmuth, Portnoy & Wright, *Journal of Semitic Studies* 67.2, 2022, pp. 441–469) and would be well suited to evaluate the statistical methodology independently of the biblical studies content.

- **Dr. Cody Kingham** (University of Notre Dame, Data Science & Hebrew Bible) — has applied computational linguistic and machine learning methods to the BHSA/ETCBC database of Biblical Hebrew, including collaborative work on probabilistic approaches to syntactic variation in Biblical Hebrew. He is perhaps uniquely positioned to evaluate both the corpus methodology and the statistical framework.

- **Prof. Wido van Peursen** (Vrije Universiteit Amsterdam) — a principal contributor to the BHSA/ETCBC corpus on which this paper's Hebrew data extraction depends. He would be well placed to assess whether the feature extraction pipeline is consistent with how the corpus is designed to be used.

- **Dr. Martijn Naaijer** (University of Zurich) — has engaged critically with existing linguistic dating methods and their statistical assumptions (Rezetko, Young, Ehrensvärd & Naaijer, *Bible and Interpretation*, 2025), making him well suited to evaluate whether the present framework addresses the methodological objections he and his collaborators have raised.

- **Prof. Jacob L. Wright** (Emory University, Candler School of Theology) — a scholar of Hebrew Bible whose recent work (*Why the Bible Began*, 2023) engages seriously with the historical and compositional questions the paper's results bear on. He would be able to assess whether the biblical studies framing and the Discussion of the Documentary Hypothesis results are handled appropriately.

I would ask that the paper not be sent to scholars whose public positions make a methodologically focused review unlikely. In particular, I would prefer to exclude **Prof. Aaron D. Hornkohl** (University of Cambridge), who has recently defended existing linguistic dating methodology in terms that suggest he may have difficulty engaging with a framework that questions the method's core assumptions, and whose response to critics of the field (Hornkohl, *Bible and Interpretation*, 2026) does not suggest he is approaching these questions as open empirical ones.

Thank you for your time and consideration.

Sincerely,

Aaron Adair
Research Affiliate, MIT Physics
adairaar@gmail.com
