"""
03_feature_extraction.py
========================
Extracts morphosyntactic and lexical features from the preprocessed Greek corpus.

All features are theoretically motivated with documented diachronic trajectories
in the scholarship on Greek language change (Classical → Koine → Late Imperial).
No character n-grams or word bigrams are included — those track register×date
confounds rather than genuine within-register diachronic change.

CRITICAL: All token matching uses strip_diacritics() to normalise polytonic Unicode
(U+1F00–U+1FFF) to plain base letters before any regex comparison.  Academic Greek
editions use the polytonic block (e.g. "καὶ" with ι=U+1F76 varia) not the modern
monotonic block (U+03AF tonos), so accented patterns silently match nothing.

Feature inventory
-----------------
A. Morphosyntactic (verbal categories)
   A01  opt_rate         — Optative mood (per 1k tokens); ↓ Classical→Koine
   A02  inf_rate         — Infinitive (per 1k tokens); ↓ Koine (replaced by ἵνα/ὅτι)
   A03  hina_rate        — ἵνα-clause (per 1k tokens); ↑ replaces infinitive in Koine
   A04  hoti_rate        — ὅτι/ὡς complementizer (per 1k tokens); ↑ Koine
   A05  part_rate        — Participial forms (per 1k tokens)
   A06  dual_rate        — REMOVED. Dual forms cannot be reliably detected with simple suffix
                           matching: -τον matches αὐτόν/θάνατον, -τοιν and -εσθον give counts
                           < 0.2/1k even in Plato, below useful signal level. A morphological
                           tagger (e.g., CLTK) would be required for reliable dual detection.

B. Particle and conjunction rates
   B01  men_rate         — μέν (per 1k tokens); ↓ Koine
   B02  de_rate          — δέ (per 1k tokens)
   B03  men_de_ratio     — μέν/(μέν+δέ); ↓ in Koine
   B04  an_rate          — ἄν particle (per 1k tokens); ↓ Koine
   B05  gar_rate         — γάρ (per 1k tokens); ↓ Koine
   B06  oun_rate         — οὖν (per 1k tokens); ↑ Koine
   B07  ean_rate         — ἐάν (per 1k tokens); ↑ Koine (conditional particle)
   B08  goun_toinun_rate — γοῦν / τοίνυν (per 1k tokens); ↓ Classical connectives

C. Pronoun, preposition, and negation shifts
   C01  autos_pron_rate  — αὐτός nominative (per 1k tokens); ↑ as 3rd-person pronoun Koine
   C02  kai_rate         — καί (per 1k tokens); ↑ Koine and LXX
   C03  en_rate          — ἐν (per 1k tokens)
   C04  eis_rate         — εἰς (per 1k tokens); ↑ Koine (encroaches on ἐν for location)
   C05  prep_en_eis_frac — ἐν/(ἐν+εἰς); ↓ Koine
   C06  ou_rate          — οὐ/οὐκ/οὐχ negation (per 1k tokens)
   C07  me_rate          — μή negation (per 1k tokens)
   C08  ou_me_frac       — οὐ/(οὐ+μή)

D. Orthographic
   D01  tt_ss_ratio      — -ττ-/(−ττ-+−σσ-) token fraction; Attic -ττ- → Koine -σσ-

E. Discourse-level
   E01  avg_sent_len     — Mean tokens per sentence; ↓ Koine
   E02  kai_sent_initial — Fraction of sentences beginning with καί; ↑ LXX (waw-consec.)
   E03  type_token_ratio — Unique types / total tokens (lexical diversity)

F. LXX / Semitic-substrate register markers
   F01  idou_rate        — ἰδού (per 1k tokens); proxy for Hebrew הִנֵּה
   F02  eipen_rate       — εἶπεν / εἶπον narrative formula (per 1k tokens)

Output
------
   data/features/grammatical_features.csv  — all feature rates, one row per entry
   data/features/feature_matrix.csv        — alias (read by 04_feature_screening.py)

Usage
-----
    python 03_feature_extraction.py [--rebuild]
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("pandas not installed. Run: pip install pandas --break-system-packages")
    sys.exit(1)

HERE     = os.path.dirname(os.path.abspath(__file__))
PROC_DIR = os.path.join(HERE, "data", "processed")
FEAT_DIR = os.path.join(HERE, "data", "features")
MANIFEST = os.path.join(HERE, "corpus_manifest.json")


# ---------------------------------------------------------------------------
# Unicode normalisation — MUST be applied before any pattern matching
# ---------------------------------------------------------------------------

def strip_diacritics(s: str) -> str:
    """
    Remove all combining diacritical marks from a Greek string, converting
    polytonic Unicode (U+1F00–U+1FFF) to plain base letters.

    Examples:
      "καὶ"  (ι=U+1F76 iota-with-varia)  → "και"
      "μέν"  (έ=U+03AD)                  → "μεν"
      "ἵνα"  (ἵ=U+1F35)                  → "ινα"
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


# ---------------------------------------------------------------------------
# Regex patterns — ALL written in stripped / unaccented form.
# Match against strip_diacritics(token), not the raw token.
# ---------------------------------------------------------------------------

# A01: Optative mood endings.
# Classical optative is already rare by 3rd century BCE; essentially absent in Koine.
OPT_PAT = re.compile(
    r"\w+(?:"
    r"οιμι|οιης|οιη|οιμεν|οιτε|οιεν|"        # thematic pres / 2nd-aor opt
    r"αιμι|αιης|αιη|αιμεν|αιτε|αιεν|"        # sigmatic aor opt (λύσαιμι)
    r"σαιμι|σαιης|σαιη|σαιμεν|σαιτε|σαιεν|"  # -σα- aor opt
    r"ειην|ειης|ειη|ειμεν|ειτε|ειεν|"        # athematic opt (ειην κτλ.)
    r"θειην|θειης|θειη|θειμεν|θειτε|θειεν"   # θη-aor passive opt
    r")$",
    re.UNICODE,
)

# A02: Infinitive endings (unaccented).
# Redundant sub-patterns removed (-αναι ⊂ -ναι; -εσθαι ⊂ -σθαι; -εσεσθαι ⊂ -σεσθαι).
INF_PAT = re.compile(
    r"\w+(?:"
    r"ειν|"       # pres/2nd-aor act thematic (λύειν→λυειν, εἶναι stem...)
    r"ναι|"       # athematic pres inf (εἶναι→ειναι, ἱέναι→ιεναι, διδόναι→διδοναι)
    r"σαι|"       # weak aor act inf (λύσαι→λυσαι)
    r"σθαι|"      # mid/pass inf (λύεσθαι→λυεσθαι, λύσασθαι→λυσασθαι, etc.)
    r"θηναι|"     # aor pass inf (λυθηναι)
    r"σεσθαι"     # fut mid inf (λύσεσθαι→λυσεσθαι)
    r")$",
    re.UNICODE,
)

# A03: ινα — stripped from ἵνα (purpose / complementizer; ↑ in Koine)
HINA_PAT = re.compile(r"^ινα$", re.UNICODE)

# A04: οτι / ως as complementizer — stripped from ὅτι / ὡς
HOTI_PAT = re.compile(r"^(?:οτι|ως)$", re.UNICODE)

# A05: Participial endings (unaccented).
# Longer, more specific endings reduce false positives from common noun/adj forms.
PART_PAT = re.compile(
    r"\w+(?:"
    # Present active
    r"ουσα|ουσης|ουσι|ουσαν|"
    r"οντος|οντι|οντα|οντες|οντων|"
    # Present middle/passive
    r"ομενος|ομενη|ομενον|ομενου|ομενης|ομενοι|ομεναι|ομενων|"
    # Aorist active (weak: -αντ-, -ασ-)
    r"ασα|αντος|αντι|αντα|αντες|αντων|"
    # Aorist passive (-εντ-)
    r"εισα|εντος|εντι|εντα|εντες|εντων|"
    # Perfect middle/passive
    r"μενος|μενη|μενον|μενου|μενης|μενοι|μεναι|μενων"
    r")$",
    re.UNICODE,
)

# NOTE: dual_rate removed. Dual verbal endings overlap fatally with common words
# (-ατον: εκατον/θανατον; -ετην: αρετην/νομοθετην; -τοιν gives < 0.02/1k even in Plato).
# Reliable dual detection requires a full morphological tagger (e.g. CLTK).

# B01: μεν — stripped from μέν
MEN_PAT = re.compile(r"^μεν$", re.UNICODE)

# B02: δε — stripped from δέ
DE_PAT = re.compile(r"^δε$", re.UNICODE)

# B04: αν — stripped from ἄν
AN_PAT = re.compile(r"^αν$", re.UNICODE)

# B05: γαρ — stripped from γάρ
GAR_PAT = re.compile(r"^γαρ$", re.UNICODE)

# B06: ουν — stripped from οὖν
OUN_PAT = re.compile(r"^ουν$", re.UNICODE)

# B07: εαν — stripped from ἐάν
EAN_PAT = re.compile(r"^εαν$", re.UNICODE)

# B08: γουν / τοινυν — stripped from γοῦν / τοίνυν
GOUN_PAT = re.compile(r"^(?:γουν|τοινυν)$", re.UNICODE)

# C01: αυτος etc. in nominative — stripped from αὐτός/αὐτή/αὐτό/αὐτοί/αὐταί/αὐτά
AUTOS_NOM_PAT = re.compile(r"^(?:αυτος|αυτη|αυτο|αυτοι|αυται|αυτα)$", re.UNICODE)

# C02: και — stripped from καί
KAI_PAT = re.compile(r"^και$", re.UNICODE)

# C03/C04: εν / εις — stripped from ἐν / εἰς
EN_PAT  = re.compile(r"^εν$",  re.UNICODE)
EIS_PAT = re.compile(r"^εις$", re.UNICODE)

# C06/C07: negation
OU_PAT = re.compile(r"^(?:ου|ουκ|ουχ|ουχι)$", re.UNICODE)
ME_PAT = re.compile(r"^(?:μη|μηδε|μηδεις|μητε)$", re.UNICODE)

# D01: -ττ- / -σσ- consonant clusters (unaffected by diacritics; strip anyway for safety)
TT_PAT = re.compile(r"ττ", re.UNICODE)
SS_PAT = re.compile(r"σσ", re.UNICODE)

# F01: ιδου — stripped from ἰδού
IDOU_PAT = re.compile(r"^ιδου$", re.UNICODE)

# F02: ειπεν / ειπον — stripped from εἶπεν / εἶπον
EIPEN_PAT = re.compile(r"^(?:ειπεν|ειπον)$", re.UNICODE)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_text(eid: str) -> list[list[str]]:
    """Load tokenized text for a corpus entry.  Returns list of sentences."""
    path = os.path.join(PROC_DIR, f"{eid}.txt")
    if not os.path.exists(path):
        return []
    sentences = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            toks = line.strip().split()
            if toks:
                sentences.append(toks)
    return sentences


def load_meta(eid: str) -> dict:
    path = os.path.join(PROC_DIR, f"{eid}_meta.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_ZERO_KEYS = [
    "opt_rate", "inf_rate", "hina_rate", "hoti_rate", "part_rate",
    "men_rate", "de_rate", "men_de_ratio",
    "an_rate", "gar_rate", "oun_rate", "ean_rate", "goun_toinun_rate",
    "autos_pron_rate", "kai_rate",
    "en_rate", "eis_rate", "prep_en_eis_frac",
    "ou_rate", "me_rate", "ou_me_frac",
    "tt_ss_ratio",
    "avg_sent_len", "kai_sent_initial", "type_token_ratio",
    "idou_rate", "eipen_rate",
]


def extract_features(sentences: list[list[str]]) -> dict:
    """
    Extract all morphosyntactic and lexical features from a tokenized text.

    strip_diacritics() is applied to every token before any pattern matching
    so that polytonic academic editions (U+1F00–U+1FFF) are handled correctly.
    """
    tokens_raw = [w for s in sentences for w in s]
    n = len(tokens_raw)

    if n == 0:
        return {k: 0.0 for k in _ZERO_KEYS}

    # ── Strip diacritics once; all subsequent matching uses `stripped` ────────
    stripped = [strip_diacritics(w) for w in tokens_raw]

    per1k = 1000.0 / n

    # A: Morphosyntactic
    opt  = sum(1 for w in stripped if OPT_PAT.match(w))
    inf  = sum(1 for w in stripped if INF_PAT.match(w))
    hina = sum(1 for w in stripped if HINA_PAT.match(w))
    hoti = sum(1 for w in stripped if HOTI_PAT.match(w))
    part = sum(1 for w in stripped if PART_PAT.match(w))

    # B: Particles
    men  = sum(1 for w in stripped if MEN_PAT.match(w))
    de   = sum(1 for w in stripped if DE_PAT.match(w))
    an   = sum(1 for w in stripped if AN_PAT.match(w))
    gar  = sum(1 for w in stripped if GAR_PAT.match(w))
    oun  = sum(1 for w in stripped if OUN_PAT.match(w))
    ean  = sum(1 for w in stripped if EAN_PAT.match(w))
    goun = sum(1 for w in stripped if GOUN_PAT.match(w))

    # C: Pronouns / prepositions / negation
    autos = sum(1 for w in stripped if AUTOS_NOM_PAT.match(w))
    kai   = sum(1 for w in stripped if KAI_PAT.match(w))
    en    = sum(1 for w in stripped if EN_PAT.match(w))
    eis   = sum(1 for w in stripped if EIS_PAT.match(w))
    ou    = sum(1 for w in stripped if OU_PAT.match(w))
    me    = sum(1 for w in stripped if ME_PAT.match(w))

    # D: Orthographic cluster counts
    tt_cnt = sum(1 for w in stripped if TT_PAT.search(w))
    ss_cnt = sum(1 for w in stripped if SS_PAT.search(w))

    # E: Discourse features
    sent_lens    = [len(s) for s in sentences if s]
    avg_sent_len = float(np.mean(sent_lens)) if sent_lens else 0.0

    n_sents = len(sentences)
    kai_initial = sum(
        1 for s in sentences
        if s and KAI_PAT.match(strip_diacritics(s[0]))
    )
    kai_sent_initial = kai_initial / n_sents if n_sents > 0 else 0.0

    type_token_ratio = len(set(stripped)) / n

    # F: LXX / Semitic register
    idou  = sum(1 for w in stripped if IDOU_PAT.match(w))
    eipen = sum(1 for w in stripped if EIPEN_PAT.match(w))

    def frac(a: int, b: int) -> float:
        return a / (a + b) if (a + b) > 0 else float("nan")

    return {
        # A
        "opt_rate"          : opt   * per1k,
        "inf_rate"          : inf   * per1k,
        "hina_rate"         : hina  * per1k,
        "hoti_rate"         : hoti  * per1k,
        "part_rate"         : part  * per1k,
        # B
        "men_rate"          : men   * per1k,
        "de_rate"           : de    * per1k,
        "men_de_ratio"      : frac(men, de),
        "an_rate"           : an    * per1k,
        "gar_rate"          : gar   * per1k,
        "oun_rate"          : oun   * per1k,
        "ean_rate"          : ean   * per1k,
        "goun_toinun_rate"  : goun  * per1k,
        # C
        "autos_pron_rate"   : autos * per1k,
        "kai_rate"          : kai   * per1k,
        "en_rate"           : en    * per1k,
        "eis_rate"          : eis   * per1k,
        "prep_en_eis_frac"  : frac(en, eis),
        "ou_rate"           : ou    * per1k,
        "me_rate"           : me    * per1k,
        "ou_me_frac"        : frac(ou, me),
        # D
        "tt_ss_ratio"       : frac(tt_cnt, ss_cnt),
        # E
        "avg_sent_len"      : avg_sent_len,
        "kai_sent_initial"  : kai_sent_initial,
        "type_token_ratio"  : type_token_ratio,
        # F
        "idou_rate"         : idou  * per1k,
        "eipen_rate"        : eipen * per1k,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract morphosyntactic features from the preprocessed Greek corpus."
    )
    parser.add_argument("--rebuild", action="store_true",
                        help="Recompute even if outputs already exist.")
    args = parser.parse_args()

    os.makedirs(FEAT_DIR, exist_ok=True)

    with open(MANIFEST, encoding="utf-8") as fh:
        corpus = json.load(fh)

    available = []
    for entry in corpus:
        p = os.path.join(PROC_DIR, f"{entry['id']}.txt")
        if os.path.exists(p):
            available.append(entry)
        else:
            print(f"  [SKIP] {entry['id']} — no processed text (run 02_preprocess.py)")

    print(f"\nExtracting features for {len(available)} entries …\n")

    rows = []
    for entry in available:
        sents = load_text(entry["id"])
        meta  = load_meta(entry["id"])
        feats = extract_features(sents)
        row = {
            "id"         : entry["id"],
            "author"     : entry["author"],
            "date_ce"    : entry["date_ce"],
            "date_sigma" : entry["date_sigma"],
            "genre"      : entry["genre"],
            "holdout"    : entry["holdout"],
            "word_count" : meta.get("word_count", 0),
            **feats,
        }
        rows.append(row)
        print(
            f"  {entry['id']:42s}"
            f"  opt={feats['opt_rate']:5.2f}"
            f"  men={feats['men_rate']:5.2f}"
            f"  gar={feats['gar_rate']:5.2f}"
            f"  oun={feats['oun_rate']:5.2f}"
            f"  hina={feats['hina_rate']:5.2f}"
            f"  kai={feats['kai_rate']:6.2f}"
        )

    df = pd.DataFrame(rows).set_index("id")

    gram_path = os.path.join(FEAT_DIR, "grammatical_features.csv")
    df.to_csv(gram_path)

    # feature_matrix.csv is what 04_feature_screening.py reads
    feat_path = os.path.join(FEAT_DIR, "feature_matrix.csv")
    df.to_csv(feat_path)

    n_feat = len([c for c in df.columns
                  if c not in ("author", "date_ce", "date_sigma", "genre",
                                "holdout", "word_count")])
    training_n = sum(1 for e in available if not e["holdout"])
    holdout_n  = len(available) - training_n

    print(f"\n{'─'*60}")
    print(f"Feature matrix  → {feat_path}")
    print(f"  Entries        : {df.shape[0]}  ({training_n} training, {holdout_n} holdout)")
    print(f"  Feature cols   : {n_feat}")

    # ── Summary by register ──────────────────────────────────────────────────
    # Build id→register lookup from the manifest (register ≠ genre/prose-type)
    reg_lookup = {e["id"]: e.get("register", "unknown") for e in available}
    df["register"] = df.index.map(reg_lookup)

    print(f"\n{'─'*60}")
    print(f"{'Register':18s} {'n':>3}  "
          f"{'opt':>6} {'inf':>6} {'men':>6} {'gar':>6} "
          f"{'oun':>6} {'hina':>6} {'kai':>7} {'tt_ss':>6}")
    print("─" * 80)
    for reg in ["ancient_Attic", "Atticizing", "Koine", "LXX"]:
        mask = df["register"] == reg
        if not mask.any():
            continue
        sub = df[mask]
        print(
            f"  {reg:16s} {mask.sum():>3}"
            f"  {sub['opt_rate'].mean():6.2f}"
            f"  {sub['inf_rate'].mean():6.2f}"
            f"  {sub['men_rate'].mean():6.2f}"
            f"  {sub['gar_rate'].mean():6.2f}"
            f"  {sub['oun_rate'].mean():6.2f}"
            f"  {sub['hina_rate'].mean():6.2f}"
            f"  {sub['kai_rate'].mean():7.2f}"
            f"  {sub['tt_ss_ratio'].mean():6.3f}"
        )
    # Drop the temporary register column so it doesn't leak into feature_matrix.csv
    df.drop(columns=["register"], inplace=True)


if __name__ == "__main__":
    main()
