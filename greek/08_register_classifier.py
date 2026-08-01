"""
08_register_classifier.py
=========================
Stage 1 of the two-stage register-calibrated dating pipeline.

Trains a register classifier that distinguishes four stylistic registers:

  ancient_Attic  — genuine Classical/Ionic authors (440–310 BCE)
  Atticizing     — Imperial-era authors imitating Classical style
  Koine          — natural post-Classical language development
  LXX            — translation Greek with Semitic substrate
                   (Septuagint and related Jewish-Greek texts)

Architecture (non-linear)
--------------------------
Stage A — Feature extraction
    14 hand-crafted register features (interpretable, linguistics-motivated).
    Includes 3 LXX-specific features: ἰδού rate, εἶπεν formula rate,
    and sentence-initial καί proportion.

Stage B — NMF topic model (data-driven, non-linear)
    Applied to the main feature matrix (char n-grams + bigrams from
    03_feature_extraction.py).  NMF decomposes the feature space into
    n_topics=8 latent "stylistic topics".  Each text receives a topic-weight
    vector that captures patterns linear models miss (e.g. the simultaneous
    presence of multiple markers that together signal a register).

Stage C — Random Forest classifier
    RandomForestClassifier trained on the combined (14 + 8) feature vector.
    Non-linear, resistant to overfitting on small datasets (100 trees,
    balanced class weights), no assumption of linear separability.

Why not LDA (Linear Discriminant Analysis)?
    LDA projects data onto axes that *maximally separate* class means.
    When classes have non-Gaussian or overlapping distributions — as
    registers do — LDA finds a compromise that mislocates all groups.
    In the earlier 3-class version, CV accuracy was only 53% partly
    because Atticizing and ancient_Attic overlap heavily on linear axes.
    RandomForest can exploit feature interactions (e.g. high dual_rate
    AND high goun_toinun_rate together → ancient_Attic, not Atticizing).

LXX register — distinctive features
--------------------------------------
The Septuagint is translation Greek from Hebrew/Aramaic originals.
Key linguistic fingerprints:

  ἰδού rate          — exclamatory/deictic particle (ἰδού = "behold"),
                        reflects Hebrew הִנֵּה (hinneh). Very rare in
                        Classical Greek; extremely high in LXX and NT.

  εἶπεν rate         — aorist 3sg of λέγω ("he/she said"), the standard
                        narrative formula.  In Attic prose λέγει (historic
                        present) or εἶπε are preferred.  LXX uses εἶπεν
                        with nearly formulaic frequency due to the Hebrew
                        וַיֹּאמֶר (wayyōʾmer) narrative pattern.

  καί sentence-initial — proportion of sentences beginning with καί.
                        Reflects Hebrew waw-consecutive (וַ-).  Classical
                        Attic almost never begins sentences with καί;
                        LXX does so constantly.

  ἵνα rate (shared with Koine) — Classical ἵνα introduces purpose clauses
                        only.  LXX/Koine extends it to indirect commands
                        (replacing the infinitive + ὥστε), following
                        Semitic ל- (lamed) + infinitive constructions.

  γοῦν / τοίνυν rate — strong Attic discourse particles; nearly absent
                        in LXX.

  -ττ- / -σσ- ratio   — LXX largely follows Koine (-σσ-) spellings.

Output
------
  results/register_probs.json       — P(register) dict keyed by text id
  results/register_classifier.pkl  — serialised sklearn Pipeline + NMF
  results/register_features.csv    — hand-crafted feature matrix
  results/register_lxx_diagnostic.txt — LXX vs Classical feature analysis
  results/plots/register_nmf.png   — NMF topic biplot
  results/plots/register_lxx_bars.png — LXX diagnostic bar chart

Usage
-----
    python 08_register_classifier.py
"""

import json
import os
import pickle
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.decomposition import NMF
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

HERE       = os.path.dirname(os.path.abspath(__file__))
PROC_DIR   = os.path.join(HERE, "data", "processed")
FEAT_DIR   = os.path.join(HERE, "data", "features")
RESULTS    = os.path.join(HERE, "results")
PLOTS_DIR  = os.path.join(HERE, "results", "plots")
MANIFEST   = os.path.join(HERE, "corpus_manifest.json")

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# All four registers (LXX added)
KNOWN_REGISTERS = ["ancient_Attic", "Atticizing", "Koine", "LXX"]

# NMF: number of latent stylistic topics
N_TOPICS = 8


# ---------------------------------------------------------------------------
# Greek text helpers
# ---------------------------------------------------------------------------

def load_text(eid: str) -> str:
    path = os.path.join(PROC_DIR, f"{eid}.txt")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def tokenize(text: str) -> list[str]:
    """Split on whitespace, strip punctuation, lowercase, normalise unicode."""
    text = unicodedata.normalize("NFC", text)
    tokens = re.split(r"\s+", text.strip())
    cleaned = []
    for tok in tokens:
        tok = re.sub(r"^[·.,;·:?!\"'()\[\]{}<>«»—–\-]+", "", tok)
        tok = re.sub(r"[·.,;·:?!\"'()\[\]{}<>«»—–\-]+$", "", tok)
        if tok:
            cleaned.append(tok.lower())
    return cleaned


def strip_diacritics(word: str) -> str:
    """Remove all Greek diacritics; keep base letters."""
    nfkd = unicodedata.normalize("NFD", word)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


# ---------------------------------------------------------------------------
# Hand-crafted feature extractors
# ---------------------------------------------------------------------------
#
# IMPORTANT — Unicode normalisation
# ---------------------------------
# Academic Greek editions use POLYTONIC spelling with full diacritics from
# the precomposed polytonic block (U+1F00–U+1FFF).  The word "and" appears
# as "καὶ" (ι = U+1F76, iota with varia) not "καί" (ί = U+03AF, tonos).
# Regex patterns that embed accented characters therefore ONLY match if they
# use the identical code-point.  To avoid silent mismatches we normalise
# ALL tokens and pattern targets to their UNACCENTED base form (strip_diacritics)
# before matching.  This also unifies monotonic / polytonic variation.

# Dual-number token patterns — matched against diacritic-stripped tokens
_DUAL_PATTERNS = re.compile(
    r"\b("
    r"δυ[οω]|δυοι[ν]?|δυει[ν]?"           # δύο family (stripped)
    r"|τω|τοιν|ταιν"                         # article duals
    r"|αυτω|αυτοιν|αυταιν"                   # αὐτός duals
    r"|λεγετον|λεγοιτον|φερετον"             # verb 2nd dual
    r")\b"
)

# Attic/Koine discourse particles — UNACCENTED patterns
_GOUN   = re.compile(r"\bγουν\b")           # γοῦν (stripped)
_TOINUN = re.compile(r"\bτοινυν\b")         # τοίνυν (stripped)
_HINA   = re.compile(r"\bινα\b")            # ἵνα (stripped)
_MEN    = re.compile(r"\bμεν\b")            # μέν (stripped)
_OUN    = re.compile(r"\bουν\b")            # οὖν (stripped)
_KAI    = re.compile(r"\bκαι\b")            # καί / καὶ (stripped → και)
_DE     = re.compile(r"\bδε\b")             # δέ (stripped)

# LXX-specific markers — UNACCENTED
_IDOU  = re.compile(r"\bιδου\b")            # ἰδού (stripped)
_EIPEN = re.compile(r"\bειπ[εα]ν?\b")      # εἶπεν/εἶπε/εἶπαν (stripped)

# -ττ- vs -σσ- variants (no diacritics on consonant clusters — unaffected)
_TT = re.compile(r"ττ")
_SS = re.compile(r"σσ")

# Optative mood heuristic suffixes — UNACCENTED (all already bare vowels)
_OPT_SUFFIXES = re.compile(
    r"(οιμι|οις|οι|οιμεν|οιτε|οιεν"
    r"|αιμι|αις|αι|αιμεν|αιτε|αιεν"
    r"|ειμι|εις|ει|ειμεν|ειτε|ειεν"
    r"|ειην|ειης|ειη|ειημεν|ειητε|ειησαν"
    r"|σαιμι|σαις|σαι|σαιμεν|σαιτε|σαιεν"
    r"|θειην|θειης|θειη|θειεν)$"
)

# Participle suffix heuristic — unaccented vowels (already base forms)
_PTCP_SUFFIXES = re.compile(
    r"(ων|ουσα|ον|οντα|οντος|ουσης|ουσαν"
    r"|ας|ασα|αν|αντα|αντος|ασης"
    r"|εις|εισα|εν|εντα|εντος|εισης"
    r"|ωτ|ωσα|[οω]ν)$"
)

_FEATURE_NAMES = [
    "dual_rate",        "goun_toinun_rate",  "hina_rate",
    "tt_ss_ratio",      "men_rate",           "oun_rate",
    "opt_rate",         "ptcp_rate",          "kai_rate",
    "de_rate",          "avg_sent_len",
    # LXX-specific
    "idou_rate",        "eipen_rate",         "kai_sent_initial",
]

# Human-readable labels for diagnostic output
_FEAT_LABELS = {
    "dual_rate":         "Dual number forms / 1k words",
    "goun_toinun_rate":  "γοῦν + τοίνυν (Attic particles) / 1k",
    "hina_rate":         "ἵνα clauses / 1k words",
    "tt_ss_ratio":       "-ττ- / (-ττ-+-σσ-) ratio  [Attic vs Koine spelling]",
    "men_rate":          "μέν / 1k words",
    "oun_rate":          "οὖν / 1k words",
    "opt_rate":          "Optative forms (heuristic) / 1k words",
    "ptcp_rate":         "Participial forms (heuristic) / 1k words",
    "kai_rate":          "καί / 1k words",
    "de_rate":           "δέ / 1k words",
    "avg_sent_len":      "Average sentence length (words)",
    "idou_rate":         "ἰδού (deictic, =Heb. hinneh) / 1k words",
    "eipen_rate":        "εἶπεν/εἶπε formula (narrative 'he said') / 1k",
    "kai_sent_initial":  "Sentences beginning with καί (waw-consecutive proxy)",
}


def extract_register_features(text: str) -> dict:
    """
    Extract 14 register-discriminating hand-crafted features.
    Rates are per 1000 words; ratios are in [0, 1].

    All regex matching is done on DIACRITIC-STRIPPED tokens to avoid
    Unicode polytonic/monotonic mismatches (e.g. καὶ vs καί).
    The -ττ-/-σσ- ratio uses the original tokens (diacritics on vowels
    don't affect consonant clusters).
    """
    if not text.strip():
        return {k: 0.0 for k in _FEATURE_NAMES}

    tokens = tokenize(text)           # lowercased, punctuation-stripped
    n = len(tokens)
    if n == 0:
        return {k: 0.0 for k in _FEATURE_NAMES}

    # Strip diacritics from every token for accent-insensitive matching
    stripped = [strip_diacritics(t) for t in tokens]

    scale = 1000.0 / n

    dual_count      = sum(1 for t in stripped if _DUAL_PATTERNS.search(t))
    goun_toinun     = sum(1 for t in stripped if _GOUN.search(t) or _TOINUN.search(t))
    hina_count      = sum(1 for t in stripped if _HINA.search(t))
    # -ττ- / -σσ- ratio: use original tokens (consonants have no diacritics)
    tt_count        = sum(1 for t in tokens   if _TT.search(t))
    ss_count        = sum(1 for t in tokens   if _SS.search(t))
    tt_ss_ratio     = tt_count / (tt_count + ss_count + 1e-9)
    men_count       = sum(1 for t in stripped if _MEN.search(t))
    oun_count       = sum(1 for t in stripped if _OUN.search(t))
    opt_count       = sum(1 for t in stripped if len(t) >= 4 and _OPT_SUFFIXES.search(t))
    ptcp_count      = sum(1 for t in stripped if len(t) >= 4 and _PTCP_SUFFIXES.search(t))
    kai_count       = sum(1 for t in stripped if _KAI.search(t))
    de_count        = sum(1 for t in stripped if _DE.search(t))
    idou_count      = sum(1 for t in stripped if _IDOU.search(t))
    eipen_count     = sum(1 for t in stripped if _EIPEN.search(t))

    # Sentence-initial καί: proportion of sentences starting with a καί token
    # Use newlines (from preprocessor) as primary sentence boundaries
    raw_text = unicodedata.normalize("NFC", text)
    sentences = re.split(r"[.;·?!\n]+", raw_text)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 2]
    kai_initial_count = 0
    for sent in sentences:
        first_tok = re.split(r"\s+", sent.strip())[0]
        first_tok = re.sub(r"^[·.,;:?!\"'()\[\]{}<>«»—–\-]+", "", first_tok)
        first_tok = re.sub(r"[·.,;:?!\"'()\[\]{}<>«»—–\-]+$", "", first_tok)
        # Strip diacritics for matching
        first_stripped = strip_diacritics(first_tok.lower())
        if _KAI.search(first_stripped):
            kai_initial_count += 1
    kai_sent_initial = kai_initial_count / (len(sentences) + 1e-9)

    # Average sentence length (words)
    avg_sent_len = (sum(len(s.split()) for s in sentences) / len(sentences)
                   if sentences else 10.0)

    return {
        "dual_rate"       : dual_count    * scale,
        "goun_toinun_rate": goun_toinun   * scale,
        "hina_rate"       : hina_count    * scale,
        "tt_ss_ratio"     : tt_ss_ratio,
        "men_rate"        : men_count     * scale,
        "oun_rate"        : oun_count     * scale,
        "opt_rate"        : opt_count     * scale,
        "ptcp_rate"       : ptcp_count    * scale,
        "kai_rate"        : kai_count     * scale,
        "de_rate"         : de_count      * scale,
        "avg_sent_len"    : avg_sent_len,
        "idou_rate"       : idou_count    * scale,
        "eipen_rate"      : eipen_count   * scale,
        "kai_sent_initial": kai_sent_initial,
    }


# ---------------------------------------------------------------------------
# NMF topic model helper
# ---------------------------------------------------------------------------

def build_nmf_features(feat_matrix_path: str, robust_names_path: str,
                        ids_ordered: list[str], n_topics: int = 8):
    """
    Load the main feature matrix, apply NMF to derive n_topics latent style
    topics, and return a DataFrame of topic weights aligned to ids_ordered.

    The NMF model decomposes X ≈ W × H where:
      W[i, k]  = weight of topic k in text i  (what we use as features)
      H[k, j]  = weight of feature j in topic k  (for interpretation)

    Returns
    -------
    topic_df : DataFrame  shape (len(ids_ordered), n_topics)
    nmf      : fitted NMF object (for inspection / diagnostics)
    feature_names : list of feature column names used
    """
    df_feat = pd.read_csv(feat_matrix_path, index_col="id")
    # Use all numeric feature columns (not metadata) — robust_names may now be very
    # small after removing n-grams, which would cause NMF n_components > n_features.
    meta_skip = {"author", "date_ce", "date_sigma", "genre", "holdout", "word_count"}
    feat_cols = [c for c in df_feat.columns if c not in meta_skip
                 and pd.api.types.is_numeric_dtype(df_feat[c])]
    # Cap n_topics to avoid NMF dimension error
    n_topics = min(n_topics, len(feat_cols) - 1)

    # Reindex to match ids_ordered; fill missing with 0
    df_sub = df_feat.reindex(ids_ordered)[feat_cols].fillna(0.0)
    X = np.maximum(df_sub.values, 0.0)   # NMF requires non-negative

    nmf = NMF(n_components=n_topics, random_state=42, max_iter=1000,
              init="nndsvda", l1_ratio=0.1)
    W = nmf.fit_transform(X)

    # Normalize each row so topic weights sum to 1 (gives interpretable proportions)
    row_sums = W.sum(axis=1, keepdims=True)
    W_norm = np.where(row_sums > 0, W / row_sums, 1.0 / n_topics)

    topic_df = pd.DataFrame(
        W_norm, index=ids_ordered,
        columns=[f"topic_{k}" for k in range(n_topics)]
    )
    return topic_df, nmf, feat_cols


# ---------------------------------------------------------------------------
# LXX diagnostic output
# ---------------------------------------------------------------------------

def _lxx_diagnostic(df_train: pd.DataFrame, feat_names: list[str],
                     topic_names: list[str], out_path: str) -> None:
    """
    Compare LXX vs. ancient_Attic and LXX vs. Koine on all features.
    Write a human-readable diagnostic table to out_path.
    """
    lxx_mask  = df_train["register"] == "LXX"
    attic_mask = df_train["register"] == "ancient_Attic"
    koine_mask = df_train["register"] == "Koine"

    all_feats = feat_names + topic_names

    lines = []
    lines.append("=" * 80)
    lines.append("LXX REGISTER — DIAGNOSTIC FEATURE ANALYSIS")
    lines.append("=" * 80)
    lines.append(
        "\nThis table shows how Septuagint Greek (LXX) differs from the other\n"
        "three registers on each feature.  Effect size = (LXX mean − other mean)\n"
        "divided by the pooled standard deviation.  |effect| > 0.8 = large.\n"
    )

    # Helper: mean and SD
    def stats(mask, feat):
        vals = df_train.loc[mask, feat].dropna().values
        return (float(np.mean(vals)) if len(vals) else 0.0,
                float(np.std(vals))  if len(vals) else 0.0)

    for comparison_name, other_mask in [
        ("LXX vs. Ancient Attic", attic_mask),
        ("LXX vs. Koine",         koine_mask),
    ]:
        lines.append(f"\n{'─'*80}")
        lines.append(f"  {comparison_name}")
        lines.append(f"  LXX texts: {lxx_mask.sum()}    "
                     f"Other texts: {other_mask.sum()}")
        lines.append(f"{'─'*80}")
        lines.append(f"  {'Feature':50s}  {'LXX mean':>10} {'Other mean':>10} "
                     f"{'Effect':>8}  Direction")
        lines.append(f"  {'─'*50}  {'─'*10} {'─'*10} {'─'*8}  {'─'*12}")

        rows = []
        for feat in all_feats:
            if feat not in df_train.columns:
                continue
            lxx_mean, lxx_sd  = stats(lxx_mask,   feat)
            oth_mean, oth_sd   = stats(other_mask, feat)
            pooled = (lxx_sd + oth_sd) / 2.0
            effect = (lxx_mean - oth_mean) / (pooled + 1e-9)
            rows.append((feat, lxx_mean, oth_mean, effect))

        # Sort by |effect|, descending
        rows.sort(key=lambda r: abs(r[3]), reverse=True)

        for feat, lxm, otm, eff in rows:
            label = _FEAT_LABELS.get(feat, feat)
            direction = ("↑ LXX higher" if eff > 0 else "↓ LXX lower")
            flag = "  *** LARGE"  if abs(eff) > 0.8 else ""
            flag = "  **  medium" if 0.5 < abs(eff) <= 0.8 else flag
            lines.append(
                f"  {label[:50]:50s}  {lxm:10.4f} {otm:10.4f} "
                f"{eff:8.3f}  {direction}{flag}"
            )

    # ── Plain-language summary ──────────────────────────────────────────────
    lines.append(f"\n\n{'='*80}")
    lines.append("PLAIN-LANGUAGE SUMMARY: HOW LXX GREEK DIFFERS FROM CLASSICAL ATTIC")
    lines.append("="*80)
    lines.append("""
What makes the Septuagint's Greek distinctive?

1.  HIGH ἰδού rate (large effect vs. Classical)
    ἰδού ("behold!") is rare in Classical prose — it sounds theatrical.
    In the LXX it appears constantly, translating Hebrew הִנֵּה (hinneh),
    the standard deictic/presentative particle.  It is also very frequent
    in the NT Gospels (esp. Matthew and Luke), a diagnostic link to the
    LXX tradition.

2.  HIGH εἶπεν rate (large effect vs. Classical)
    Classical narrative uses the historic present (λέγει) or εἶπε.
    The LXX uses εἶπεν almost formulaically as the standard 3sg aorist
    of the speech-introduction formula ("and he said..."), reproducing
    Hebrew וַיֹּאמֶר (wayyōʾmer).  This creates a distinctive rhythmic
    repetition absent from literary Koine like Josephus or Philo.

3.  HIGH sentence-initial καί (waw-consecutive proxy)
    Hebrew narrative is structured by the waw-consecutive construction
    (ו + prefixed verb), which the LXX translates almost mechanically
    as καί + verb at the start of each clause.  This is the most
    distinctive syntactic feature of LXX Greek: sentences begin with
    καί at a rate far higher than any other register.

4.  LOW γοῦν / τοίνυν (Attic particles absent)
    These are sophisticated discourse particles of educated Attic prose.
    They are essentially absent from the LXX — translation Greek operates
    at a different register of formality than rhetorical prose.

5.  LOW -ττ- ratio (follows Koine spelling)
    The LXX was translated in Alexandria (Egypt) and follows the Koine
    -σσ- spellings (θάλασσα, πράσσω) rather than Attic -ττ-.

6.  HIGH ἵνα rate (shared with Koine; even higher in LXX)
    LXX generalises ἵνα as an all-purpose subordinator, reflecting
    the Hebrew ל (lamed) + infinitive construction. This is a key feature
    distinguishing LXX from Classical Greek, where ἵνα is restricted to
    final (purpose) clauses.

7.  MODERATE κaί rate (higher than Classical, but see sentence-initial)
    The high overall καί frequency is partly the waw-consecutive effect
    and partly the LXX's general preference for paratactic coordination
    over the hypotactic (subordinate-clause-heavy) style of Classical Attic.

Implications for NT dating
--------------------------
The NT Gospels (Mark, Matthew, Luke) show a profile intermediate between
Koine and LXX: high ἰδού (especially Matthew), moderate εἶπεν, high
ἵνα rate, low -ττ-, low Attic particles.  The LXX register class allows
the calibrated dating model to recognise this Semitic-substrate style and
apply the appropriate sub-model rather than forcing the Gospels into either
the pure Koine or the LXX translation-Greek bucket.
""")

    text_out = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text_out)
    print(f"\nLXX diagnostic       → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not HAS_SKLEARN:
        print("ERROR: scikit-learn not installed.")
        return

    # Load manifest
    with open(MANIFEST, encoding="utf-8") as f:
        corpus = json.load(f)

    print("=" * 72)
    print("REGISTER CLASSIFIER — FEATURE EXTRACTION")
    print("=" * 72)

    # ── Extract hand-crafted features ───────────────────────────────────────
    records = []
    for entry in corpus:
        eid      = entry["id"]
        register = entry.get("register", "unknown")
        holdout  = entry.get("holdout", False)
        text     = load_text(eid)

        if not text:
            print(f"  [SKIP] {eid}: no processed text found")
            continue

        feats = extract_register_features(text)
        row   = {"id": eid, "register": register, "holdout": holdout,
                 "date_ce": entry["date_ce"], "author": entry["author"]}
        row.update(feats)
        records.append(row)

    df = pd.DataFrame(records).set_index("id")
    feat_path = os.path.join(RESULTS, "register_features.csv")
    df.reset_index().to_csv(feat_path, index=False)
    print(f"\nHand-crafted features ({len(df)} texts × {len(_FEATURE_NAMES)} features)")
    print(f"  → {feat_path}")

    # ── NMF topic model ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"NMF TOPIC MODEL  (k={N_TOPICS} topics from main feature matrix)")
    print("=" * 72)

    feat_matrix_path  = os.path.join(FEAT_DIR, "feature_matrix.csv")
    robust_names_path = os.path.join(RESULTS, "robust_feature_names.json")

    ids_ordered = df.index.tolist()
    topic_df, nmf_model, nmf_feat_cols = build_nmf_features(
        feat_matrix_path, robust_names_path, ids_ordered, n_topics=N_TOPICS
    )
    topic_names = list(topic_df.columns)

    print(f"  NMF fitted on {len(nmf_feat_cols)} robust features → {N_TOPICS} topics.")
    print(f"  Reconstruction error: {nmf_model.reconstruction_err_:.4f}")

    # Show top features per topic (H matrix)
    H = nmf_model.components_        # shape (n_topics, n_features)
    print("\n  Top 5 features per topic:")
    for k in range(N_TOPICS):
        top_idx = np.argsort(H[k])[::-1][:5]
        top_feats = [nmf_feat_cols[i] for i in top_idx]
        print(f"    topic_{k}: {', '.join(top_feats)}")

    # Merge topic features into df
    df = df.join(topic_df)

    # ── Train/test split ─────────────────────────────────────────────────────
    train_mask = (~df["holdout"]) & (df["register"].isin(KNOWN_REGISTERS))
    df_train   = df[train_mask].copy()

    print("\n" + "=" * 72)
    print("TRAINING RANDOM FOREST REGISTER CLASSIFIER  (4 classes)")
    print("=" * 72)

    print(f"\nTraining set: {len(df_train)} texts")
    for reg in KNOWN_REGISTERS:
        n = (df_train["register"] == reg).sum()
        print(f"  {reg:15s}: {n} texts")

    # ── Mean feature values by register ─────────────────────────────────────
    print("\nMean hand-crafted feature values by register (training texts):")
    summary = df_train.groupby("register")[_FEATURE_NAMES].mean().round(3)
    print(summary.to_string())

    # ── Fit classifier ───────────────────────────────────────────────────────
    all_feat_cols = _FEATURE_NAMES + topic_names

    X_train = df_train[all_feat_cols].values
    y_train = df_train["register"].values

    # RandomForest: balanced class weights handle the small LXX class (n=7)
    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    # Wrap in a standard scaler pipeline (RF doesn't strictly need it, but
    # it helps when features have very different scales for diagnostics)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf",     rf),
    ])
    pipe.fit(X_train, y_train)

    # ── Cross-validation ─────────────────────────────────────────────────────
    print("\nCross-validation (stratified k-fold):")
    # Use min class size as n_splits cap
    min_class = min((y_train == r).sum() for r in KNOWN_REGISTERS)
    n_splits = min(5, min_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred_cv = cross_val_predict(pipe, X_train, y_train, cv=cv)

    # Pass `labels=` to force sklearn to use KNOWN_REGISTERS order (sklearn
    # internally sorts class names, which misaligns target_names otherwise)
    present = [r for r in KNOWN_REGISTERS if r in y_train]
    print(classification_report(y_train, y_pred_cv,
                                 labels=present, target_names=present,
                                 zero_division=0))
    cm = confusion_matrix(y_train, y_pred_cv, labels=present)
    print("Confusion matrix:")
    print(pd.DataFrame(cm, index=KNOWN_REGISTERS, columns=KNOWN_REGISTERS).to_string())

    # ── Feature importances ──────────────────────────────────────────────────
    importances = pipe.named_steps["rf"].feature_importances_
    imp_df = pd.Series(importances, index=all_feat_cols).sort_values(ascending=False)
    print("\nTop 10 feature importances (Random Forest):")
    for feat, imp in imp_df.head(10).items():
        label = _FEAT_LABELS.get(feat, feat)
        print(f"  {label[:55]:55s}  {imp:.4f}")

    # ── Predict probabilities for all texts ──────────────────────────────────
    print("\n" + "=" * 72)
    print("REGISTER PROBABILITIES — ALL TEXTS")
    print("=" * 72)

    X_all   = df[all_feat_cols].values
    probs   = pipe.predict_proba(X_all)
    pred    = pipe.predict(X_all)
    classes = pipe.named_steps["rf"].classes_

    results = {}
    hdr = (f"\n{'ID':45s}  {'True':13s}  {'Pred':13s}  "
           f"{'P(attic)':8s}  {'P(attic.)':9s}  {'P(koi)':7s}  {'P(lxx)':7s}")
    print(hdr)
    print("-" * 115)

    for i, (idx, row) in enumerate(df.iterrows()):
        p = {c: float(probs[i, j]) for j, c in enumerate(classes)}
        results[idx] = {
            "register_true"    : row["register"],
            "register_pred"    : pred[i],
            "holdout"          : bool(row["holdout"]),
            "date_ce"          : int(row["date_ce"]),
            "p_ancient_Attic"  : p.get("ancient_Attic", 0.0),
            "p_Atticizing"     : p.get("Atticizing",    0.0),
            "p_Koine"          : p.get("Koine",         0.0),
            "p_LXX"            : p.get("LXX",           0.0),
        }
        flag  = " [HOLDOUT]" if row["holdout"] else ""
        match = "✓" if pred[i] == row["register"] else "✗"
        print(f"  {idx:43s}  {row['register']:13s}  "
              f"{pred[i]:13s} {match}  "
              f"{p.get('ancient_Attic',0):.3f}     "
              f"{p.get('Atticizing',0):.3f}      "
              f"{p.get('Koine',0):.3f}   "
              f"{p.get('LXX',0):.3f}{flag}")

    # ── LXX diagnostic ────────────────────────────────────────────────────────
    diag_path = os.path.join(RESULTS, "register_lxx_diagnostic.txt")
    _lxx_diagnostic(df_train, _FEATURE_NAMES, topic_names, diag_path)

    # ── Save results ──────────────────────────────────────────────────────────
    out_json = os.path.join(RESULTS, "register_probs.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Register probabilities → {out_json}")

    out_pkl = os.path.join(RESULTS, "register_classifier.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "pipeline"     : pipe,
            "nmf_model"    : nmf_model,
            "nmf_feat_cols": nmf_feat_cols,
            "hand_feat_names": _FEATURE_NAMES,
            "topic_names"  : topic_names,
            "all_feat_cols": all_feat_cols,
            "classes"      : list(classes),
        }, f)
    print(f"Classifier pickle     → {out_pkl}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if HAS_MPL:
        _plot_nmf(df, topic_names, n_topics=N_TOPICS)
        _plot_lxx_bars(df_train, _FEATURE_NAMES)

    print("\nDone.")


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

_REG_COLOURS = {
    "ancient_Attic": "#2166ac",
    "Atticizing"   : "#d6604d",
    "Koine"        : "#4dac26",
    "LXX"          : "#9b59b6",
}
_REG_MARKERS = {
    "ancient_Attic": "o",
    "Atticizing"   : "s",
    "Koine"        : "^",
    "LXX"          : "D",
}
_REG_LABELS = {
    "ancient_Attic": "Ancient Attic/Ionic",
    "Atticizing"   : "Atticizing (2nd Sophistic)",
    "Koine"        : "Koine",
    "LXX"          : "LXX (Septuagint / Semitic substrate)",
}


def _plot_nmf(df: pd.DataFrame, topic_names: list[str], n_topics: int) -> None:
    """
    2-D scatter of the first two NMF topic dimensions, coloured by register.
    """
    if n_topics < 2:
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    for reg in KNOWN_REGISTERS + ["unknown"]:
        mask = (df["register"] == reg) & (~df["holdout"])
        idx  = df[mask].index
        if len(idx) == 0:
            continue
        x = df.loc[idx, topic_names[0]].values
        y = df.loc[idx, topic_names[1]].values
        ax.scatter(x, y,
                   c=_REG_COLOURS.get(reg, "gray"),
                   marker=_REG_MARKERS.get(reg, "x"),
                   s=80, alpha=0.8,
                   label=_REG_LABELS.get(reg, reg), zorder=3)
        for xi, yi, eid in zip(x, y, idx):
            author = df.loc[eid, "author"].split()[0]
            ax.annotate(author, (xi, yi), fontsize=6.5, alpha=0.75,
                        xytext=(3, 3), textcoords="offset points")

    # Holdouts
    hold_mask = df["holdout"]
    for eid, row in df[hold_mask].iterrows():
        xi = row[topic_names[0]]
        yi = row[topic_names[1]]
        ax.scatter(xi, yi,
                   c=_REG_COLOURS.get(row["register"], "gray"),
                   marker="*", s=220,
                   edgecolors="k", linewidths=0.8, zorder=5)
        ax.annotate(f"[{row['author'].split()[0]}]", (xi, yi),
                    fontsize=6.5, style="italic", alpha=0.9,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(f"NMF topic_0 weight", fontsize=11)
    ax.set_ylabel(f"NMF topic_1 weight", fontsize=11)
    ax.set_title("NMF Topic Projection — Greek Prose Corpus (4 registers)\n"
                 "stars = holdout texts", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, "register_nmf.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"NMF plot              → {plot_path}")


def _plot_lxx_bars(df_train: pd.DataFrame, feat_names: list[str]) -> None:
    """
    Grouped bar chart showing LXX-specific feature means across registers.
    Only plots the most diagnostic LXX-adjacent features.
    """
    lxx_feats = ["idou_rate", "eipen_rate", "kai_sent_initial",
                 "hina_rate", "goun_toinun_rate", "tt_ss_ratio",
                 "kai_rate", "de_rate"]
    lxx_feats = [f for f in lxx_feats if f in df_train.columns]

    means = df_train.groupby("register")[lxx_feats].mean()
    means = means.reindex([r for r in KNOWN_REGISTERS if r in means.index])

    n_feats = len(lxx_feats)
    n_regs  = len(means)
    x = np.arange(n_feats)
    width = 0.8 / n_regs

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (reg, row) in enumerate(means.iterrows()):
        offset = (i - n_regs / 2 + 0.5) * width
        bars = ax.bar(x + offset, row.values,
                      width=width * 0.9,
                      color=_REG_COLOURS.get(reg, "gray"),
                      label=_REG_LABELS.get(reg, reg),
                      alpha=0.85)

    short_labels = [f.replace("_rate", "").replace("_", " ") for f in lxx_feats]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Mean value (rates per 1k words or ratio)", fontsize=10)
    ax.set_title("Register-distinctive features by register group\n"
                 "(focus on LXX-diagnostic features)", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.9)
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, "register_lxx_bars.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"LXX bar chart         → {plot_path}")


if __name__ == "__main__":
    main()
