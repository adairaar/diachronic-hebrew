"""
02_preprocess.py
================
Parses raw TEI XML files, normalizes Greek text, removes quoted passages,
and saves clean plain-text files for feature extraction.

Steps for each text
-------------------
1. Parse TEI XML (lxml), strip structural metadata (apparatus, notes, headers).
2. Extract prose body text (div, p, ab, seg elements).
3. Normalize Unicode to NFC; keep full polytonic diacritics (needed for
   morphological feature extraction).
4. Remove quoted passages using a two-pass heuristic:
     Pass A (structural): TEI <q>, <quote>, <cit> elements → drop.
     Pass B (lexical): sentences/clauses following attribution phrases
                       (φησί, ἔφη, εἶπεν, λέγει, γράφει, κτλ.) are
                       flagged and stripped up to the next sentence boundary.
   For entries marked quote_heavy=True the heuristic is applied more
   aggressively (larger context window, additional verbs).
5. Sentence-tokenize on Greek punctuation (·, ;, .).
6. Word-tokenize; keep only alphabetic Greek tokens (strip numerals,
   punctuation, Latin glosses).
7. Save as:
     data/processed/<id>.txt        — one sentence per line, words space-separated
     data/processed/<id>_meta.json  — word count, sentence count, quote fraction

Usage
-----
    python 02_preprocess.py [--id herodotus_histories] [--force]
"""

import argparse
import json
import os
import re
import sys
import unicodedata

try:
    from lxml import etree
except ImportError:
    print("lxml not installed. Run: pip install lxml --break-system-packages")
    sys.exit(1)

HERE       = os.path.dirname(os.path.abspath(__file__))
RAW_DIR    = os.path.join(HERE, "data", "raw")
PROC_DIR   = os.path.join(HERE, "data", "processed")
MANIFEST   = os.path.join(HERE, "corpus_manifest.json")

# ---------------------------------------------------------------------------
# TEI namespaces
# ---------------------------------------------------------------------------
NS = {
    "tei": "http://www.tei-c.org/ns/1.0",
    "xml": "http://www.w3.org/XML/1998/namespace",
}
TEI = "http://www.tei-c.org/ns/1.0"

# ---------------------------------------------------------------------------
# Greek normalization
# ---------------------------------------------------------------------------
# Map Beta Code sigmas if accidentally present; otherwise NFC suffices.
BETA_SIGMA = str.maketrans({"s": "σ", "S": "Σ"})   # only used if text is beta-code

def normalize_greek(text: str) -> str:
    """NFC-normalize polytonic Greek text."""
    return unicodedata.normalize("NFC", text)

# ---------------------------------------------------------------------------
# Quote removal heuristics
# ---------------------------------------------------------------------------

# Attribution verbs in present/imperfect/aorist, 3rd sg/pl
ATTRIBUTION_VERBS_BASIC = re.compile(
    r"\b(φησί[ν]?|φάσ[κ]ει|ἔφη|εἶπε[ν]?|λέγει|λέγουσι|γράφει|"
    r"ἱστορεῖ|διηγεῖται|μαρτυρεῖ|ποιεῖ|ᾄδει)\b",
    re.UNICODE
)
ATTRIBUTION_VERBS_HEAVY = re.compile(
    r"\b(φησί[ν]?|φάσ[κ]ει|ἔφη|εἶπε[ν]?|λέγει|λέγουσι|γράφει|"
    r"ἱστορεῖ|διηγεῖται|μαρτυρεῖ|ποιεῖ|ᾄδει|"
    r"κατὰ|παρά|κατά|ὡς φησιν|ὡς ἔφη|ὡς εἶπε|"
    r"ὅτι|ὡς|λόγος ἔχει)\b",
    re.UNICODE
)

# Sentence boundary: Greek ano teleia, Greek question mark (;), Latin period
SENT_BOUNDARY = re.compile(r"[·;.]\s*")

# Greek alphabetic characters (polytonic Unicode block U+0370–U+03FF, U+1F00–U+1FFF)
GREEK_ALPHA = re.compile(r"^[Ͱ-Ͽἀ-῿]+$", re.UNICODE)


def remove_quotes_lexical(text: str, heavy: bool = False) -> tuple[str, float]:
    """
    Heuristic quote removal at sentence level.
    Returns (cleaned_text, fraction_removed).
    """
    pattern = ATTRIBUTION_VERBS_HEAVY if heavy else ATTRIBUTION_VERBS_BASIC
    sentences = SENT_BOUNDARY.split(text)
    kept, dropped = [], 0

    i = 0
    while i < len(sentences):
        sent = sentences[i].strip()
        if not sent:
            i += 1
            continue
        if pattern.search(sent):
            # Drop this sentence and the next 1 (basic) or 2 (heavy) as likely quotes
            window = 2 if heavy else 1
            dropped += 1
            for _ in range(window):
                i += 1
                if i < len(sentences):
                    dropped += 1
        else:
            kept.append(sent)
        i += 1

    total = len(kept) + dropped
    frac  = dropped / total if total > 0 else 0.0
    # Join with newline so tokenize() can split on \n as a sentence boundary.
    # (SENT_BOUNDARY.split() consumed the original punctuation, so we supply
    # a substitute delimiter that won't appear in Greek n-grams.)
    return "\n".join(kept), frac

# ---------------------------------------------------------------------------
# TEI XML parsing
# ---------------------------------------------------------------------------
# Tags whose content we want to drop entirely (apparatus, notes, foreign glosses)
# The element AND its tail text are discarded.
DROP_TAGS = {
    # Structural metadata — always drop
    f"{{{TEI}}}note",
    f"{{{TEI}}}app",
    f"{{{TEI}}}rdg",
    f"{{{TEI}}}lem",
    f"{{{TEI}}}foreign",
    f"{{{TEI}}}bibl",
    f"{{{TEI}}}ref",
    f"{{{TEI}}}teiHeader",
    f"{{{TEI}}}head",
    f"{{{TEI}}}label",
    f"{{{TEI}}}num",
    # Embedded quotations from OTHER works — drop structurally (Pass A).
    # NOTE: <said> is NOT dropped here because in Plato and dialogues it marks
    # the primary text (speaker turns), not quotations from external sources.
    # <q>  is also kept because it often wraps the text of a dialogue speaker.
    # Lexical quote removal (Pass B) via attribution verbs handles the rest.
    f"{{{TEI}}}quote",
    f"{{{TEI}}}cit",
    f"{{{TEI}}}floatingText",
}

# Tags that are void markers: the element itself has no content we want, but
# the text *following* it (the tail) is genuine running text and must be kept.
# This is common in First1KGreek manuscripts where <lb/> and <pb/> encode
# physical line/page breaks inline in the prose, and the actual text is the tail.
# <milestone/> likewise marks section boundaries with no textual content.
VOID_TAGS = {
    f"{{{TEI}}}lb",         # line break — tail is the next line of text
    f"{{{TEI}}}pb",         # page break — tail is the text on the next page
    f"{{{TEI}}}milestone",  # section marker — tail is the next section's text
}

# Tags whose text we want to include
INCLUDE_TAGS = {
    f"{{{TEI}}}p",
    f"{{{TEI}}}ab",        # anonymous block
    f"{{{TEI}}}seg",
    f"{{{TEI}}}s",         # sentence
    f"{{{TEI}}}l",         # line (some prose uses this)
}


def extract_text_from_xml(xml_path: str) -> str:
    """
    Parse a TEI XML file and return the extracted prose body text as a string.
    Drops apparatus, notes, headers, and structural quotation elements.
    """
    try:
        tree = etree.parse(xml_path)
    except etree.XMLSyntaxError as e:
        # Try recovering from malformed XML
        parser = etree.XMLParser(recover=True)
        tree = etree.parse(xml_path, parser)

    root = tree.getroot()

    # Collect text from prose body
    chunks = []

    # Block-level tags: we insert a newline after their content so that
    # even texts with no in-sentence punctuation get sentence boundaries.
    BLOCK_TAGS = {
        f"{{{TEI}}}p", f"{{{TEI}}}ab", f"{{{TEI}}}div",
        f"{{{TEI}}}l", f"{{{TEI}}}lg", f"{{{TEI}}}s",
    }

    def walk(node, in_drop=False):
        tag = node.tag if isinstance(node.tag, str) else ""

        if tag in DROP_TAGS:
            return  # skip this subtree AND its tail

        # Void tags: skip the element and its children, but keep the tail
        # (the tail is handled by the parent loop below, not here)
        if tag in VOID_TAGS:
            return

        # Collect this node's text
        if node.text and not in_drop:
            t = node.text.strip()
            if t:
                chunks.append(t)

        for child in node:
            walk(child, in_drop)
            # Tail text follows the child closing tag.
            # - DROP_TAGS: drop the tail (it's part of the annotation context)
            # - VOID_TAGS: keep the tail (it's the running prose text)
            # - everything else: keep the tail
            child_tag = child.tag if isinstance(child.tag, str) else ""
            if child.tail and not in_drop and child_tag not in DROP_TAGS:
                t = child.tail.strip()
                if t:
                    chunks.append(t)

        # After a block-level element, insert a newline as sentence boundary
        if tag in BLOCK_TAGS and not in_drop:
            chunks.append("\n")

    walk(root)
    return " ".join(chunks)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[list[str]]:
    """
    Split text into sentences, then words. Returns list-of-lists.
    Only keeps tokens that are purely Greek alphabetic.
    """
    sentences = []
    # Split on Greek ano teleia (·), Greek question mark (;), or period
    for raw_sent in re.split(r"[·;.\n]+", text):
        raw_sent = raw_sent.strip()
        if not raw_sent:
            continue
        words = raw_sent.split()
        greek_words = [
            w.lower()
            for w in words
            if GREEK_ALPHA.match(w.strip("()[]{}\"'.,;·:"))
        ]
        if len(greek_words) >= 3:
            sentences.append(greek_words)
    return sentences


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess_entry(entry: dict, force: bool = False) -> dict:
    """
    Full preprocessing pipeline for one corpus entry.
    Returns a status/stats dict.
    """
    eid       = entry["id"]
    heavy     = entry.get("quote_heavy", False)
    xml_path  = os.path.join(RAW_DIR, f"{eid}.xml")
    out_txt   = os.path.join(PROC_DIR, f"{eid}.txt")
    out_meta  = os.path.join(PROC_DIR, f"{eid}_meta.json")

    if not os.path.exists(xml_path):
        print(f"  [SKIP] {eid}: raw XML not found at {xml_path}")
        return {"id": eid, "status": "missing_xml"}

    if os.path.exists(out_txt) and not force:
        with open(out_meta, encoding="utf-8") as f:
            meta = json.load(f)
        print(f"  [SKIP] {eid}: already processed ({meta.get('word_count',0):,} words)")
        return {"id": eid, "status": "cached", **meta}

    print(f"  [PROC] {eid} (quote_heavy={heavy}) …")

    # 1. Extract raw text from TEI
    raw_text = extract_text_from_xml(xml_path)
    raw_text = normalize_greek(raw_text)

    # 2. Pass B: lexical quote removal
    cleaned, quote_frac = remove_quotes_lexical(raw_text, heavy=heavy)

    # 3. Tokenize
    sentences = tokenize(cleaned)
    all_words = [w for sent in sentences for w in sent]

    word_count = len(all_words)
    sent_count = len(sentences)

    if word_count < 1000:
        print(f"  [WARN] {eid}: only {word_count} Greek words after cleaning — check XML.")

    # 4. Write plain text (one sentence per line)
    os.makedirs(PROC_DIR, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        for sent in sentences:
            f.write(" ".join(sent) + "\n")

    # 5. Write metadata
    meta = {
        "id"           : eid,
        "author"       : entry["author"],
        "work"         : entry["work"],
        "date_ce"      : entry["date_ce"],
        "date_sigma"   : entry["date_sigma"],
        "genre"        : entry["genre"],
        "holdout"      : entry["holdout"],
        "word_count"   : word_count,
        "sentence_count": sent_count,
        "quote_frac_removed": round(quote_frac, 4),
        "status"       : "ok",
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"         → {word_count:,} words, {sent_count:,} sentences, "
          f"{quote_frac*100:.1f}% dropped as quotes")
    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess Greek corpus texts.")
    parser.add_argument("--id",    default=None, help="Process only this corpus ID.")
    parser.add_argument("--force", action="store_true", help="Re-process even if output exists.")
    args = parser.parse_args()

    os.makedirs(PROC_DIR, exist_ok=True)

    with open(MANIFEST, encoding="utf-8") as f:
        corpus = json.load(f)

    if args.id:
        corpus = [c for c in corpus if c["id"] == args.id]
        if not corpus:
            print(f"ID '{args.id}' not found in manifest.")
            sys.exit(1)

    print(f"Preprocessing {len(corpus)} texts …\n")
    results = []
    for entry in corpus:
        r = preprocess_entry(entry, force=args.force)
        results.append(r)

    ok      = [r for r in results if r.get("status") in ("ok", "cached")]
    missing = [r for r in results if r.get("status") == "missing_xml"]
    warn    = [r for r in results if r.get("word_count", 9999) < 1000
               and r.get("status") == "ok"]

    print(f"\n{'='*60}")
    print(f"Processed OK     : {len(ok)}")
    print(f"Missing XML      : {len(missing)}")
    if missing:
        for r in missing:
            print(f"  {r['id']}")
    if warn:
        print(f"Low word count (<1000):")
        for r in warn:
            print(f"  {r['id']}: {r.get('word_count',0)} words")

    # Write summary
    summary_path = os.path.join(PROC_DIR, "preprocessing_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
