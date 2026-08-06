#!/usr/bin/env python3
"""
Feature Extraction from ETCBC Hebrew Bible Data
================================================
Extracts morphosyntactic features per book using the BHSA dataset via the
text-fabric library.

SETUP (one-time)
----------------
1. Install text-fabric with the GitHub backend so it can download BHSA data:

       pip install text-fabric[github]
       # or:  pip install 'text-fabric[github]'   (zsh requires quotes)

2. Download the BHSA data (~300 MB) by running this once:

       python3 -c "from tf.app import use; use('etcbc/bhsa', hoist=False)"

   Data lands in ~/text-fabric-data/github/etcbc/bhsa/tf/c by default.
   Alternatively, git-clone the BHSA repo and point --data-path at the tf/c dir:

       git clone --depth 1 https://github.com/etcbc/bhsa.git
       python3 01_feature_extraction_etcbc.py --data-path ./bhsa/tf/c

3. Run this script:

       python3 01_feature_extraction_etcbc.py

Output
------
features_by_book.csv — one row per dated book/section with raw counts + rates.
Consumed by:  02_bayesian_feature_analysis.py
              03_pca_stylometric.py
              04_sliding_window_analysis.py
"""

import sys
import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Default data location — auto-detected from text-fabric standard paths
# ---------------------------------------------------------------------------
# text-fabric downloads BHSA to different sub-paths depending on version:
#   ~/text-fabric-data/github/ETCBC/bhsa/tf/2021   (current default)
#   ~/text-fabric-data/github/etcbc/bhsa/tf/c       (older convention)
# We search all known candidates and use the first one that exists.

def _find_bhsa_path():
    base = Path.home() / 'text-fabric-data' / 'github'
    candidates = [
        base / 'ETCBC' / 'bhsa' / 'tf' / '2021',
        base / 'ETCBC' / 'bhsa' / 'tf' / 'c',
        base / 'etcbc' / 'bhsa' / 'tf' / '2021',
        base / 'etcbc' / 'bhsa' / 'tf' / 'c',
    ]
    for p in candidates:
        if p.exists() and (p / 'otype.tf').exists():
            return p
    # Fall back to the canonical 2021 path even if it doesn't exist yet
    return candidates[0]

DEFAULT_BHSA_PATH = _find_bhsa_path()


# ---------------------------------------------------------------------------
# Dated book/section specifications
# ---------------------------------------------------------------------------
# Format: (unit_name, [ETCBC_book_names], chapter_range_or_None,
#          date_BCE, date_sigma, period_label)
#
# Isaiah is split into its three commonly-recognised compositional layers.
# Chronicles (1 + 2) is treated as a single authorial corpus.
#
# Date sources: Hurvitz (1972), Young/Rezetko/Ehrensvärd (2008),
#               Hendel & Joosten (2018).

DATED_BOOK_SPECS = [
    ('Amos',         ['Amos'],                       None,       760, 15,  'pre-exilic'),
    ('Hosea',        ['Hosea'],                      None,       725, 20,  'pre-exilic'),
    ('Micah',        ['Micah'],                      None,       720, 20,  'pre-exilic'),
    ('Isaiah_1',     ['Isaiah'],                     (1,  39),   700, 15,  'pre-exilic'),
    ('Zephaniah',    ['Zephaniah'],                  None,       630, 15,  'pre-exilic'),
    ('Nahum',        ['Nahum'],                      None,       620, 20,  'pre-exilic'),
    ('Habakkuk',     ['Habakkuk'],                   None,       605, 20,  'pre-exilic'),
    ('Jeremiah',     ['Jeremiah'],                   None,       590, 15,  'pre-exilic'),
    ('Lamentations', ['Lamentations'],               None,       586, 20,  'exilic'),
    ('Ezekiel',      ['Ezekiel'],                    None,       570, 15,  'exilic'),
    ('Isaiah_2',     ['Isaiah'],                     (40, 55),   550, 20,  'exilic'),
    ('Haggai',       ['Haggai'],                     None,       520,  5,  'post-exilic'),
    ('Zechariah_1',  ['Zechariah'],                  (1,   8),   518,  5,  'post-exilic'),
    # Isaiah_3: revised to 450 BCE with wider sigma; some scholars argue Hellenistic
    # editing of parts; 450 is a conservative late-Persian midpoint.
    ('Isaiah_3',     ['Isaiah'],                     (56, 66),   450, 100, 'post-exilic'),
    ('Malachi',      ['Malachi'],                    None,       460, 20,  'post-exilic'),
    ('Jonah',        ['Jonah'],                      None,       400, 50,  'post-exilic'),
    ('Chronicles',   ['1_Chronicles', '2_Chronicles'],  None,    350, 50,  'post-exilic'),
    ('Esther',       ['Esther'],                     None,       350, 50,  'post-exilic'),
    ('Ecclesiastes', ['Ecclesiastes'],               None,       330, 80,  'post-exilic'),
    # Ezra/Nehemiah: if they presuppose the Torah in final form, they must postdate
    # it; scholars increasingly argue for 4th-century or even Hellenistic redaction.
    ('Ezra',         ['Ezra'],                       None,       350, 75,  'post-exilic'),
    ('Nehemiah',     ['Nehemiah'],                   None,       350, 75,  'post-exilic'),
    # Daniel: chapters 2–7 are Aramaic; only Hebrew chapters (1, 8–12) included.
    ('Daniel',       ['Daniel'],                     [(1,1),(8,12)], 167, 10, 'Hellenistic'),
]


# ---------------------------------------------------------------------------
# BHSA feature / lexeme codes
# ---------------------------------------------------------------------------
# All ETCBC lexeme codes use a consonantal ASCII transliteration.
# vt (verb tense) values in BHSA:
#   wayq = wayyiqtol   perf = qatal (perfect)   impf = yiqtol (imperfect)
#   ptca = active participle   ptcp = passive participle
#   infa = infinitive absolute   infc = infinitive construct
#   impv = imperative

LEX_ANOCHI  = '>NKJ'   # אנכי — archaic/CBH first-person singular pronoun
LEX_ANI     = '>NJ'    # אני  — standard first-person singular pronoun
                       # NB: BHSA uses J for yod, so אני = >NJ, NOT >NH (which = אנה)
LEX_ASHER   = '>CR'    # אשר  — CBH relative / subordinating particle
LEX_SHE     = 'C'      # ש-   — LBH/colloquial relative particle
LEX_LO      = 'L>'     # לא   — standard negator
LEX_EIN     = '>JN/'   # אין  — negative existential; trailing / = noun-class in BHSA
LEX_YESH    = 'JC/'    # יש   — positive existential; trailing / = noun-class in BHSA

VT_WAYYIQTOL  = 'wayq'
VT_QATAL      = 'perf'
VT_YIQTOL     = 'impf'
VT_PTCA       = 'ptca'
VT_PTCP       = 'ptcp'
VT_INF_ABS    = 'infa'
VT_INF_CON    = 'infc'


# ---------------------------------------------------------------------------
# Feature extraction from a list of word nodes
# ---------------------------------------------------------------------------

def extract_features(word_nodes, F):
    """
    Count linguistic features across a list of TF word nodes.

    Parameters
    ----------
    word_nodes : iterable of int
        Text-Fabric word node numbers.
    F : tf.core.api.NodeFeatures
        The TF feature object (api.F).

    Returns
    -------
    dict of str → int
    """
    c = dict(
        pronoun_anochi=0, pronoun_ani=0,
        rel_asher=0, rel_she=0,
        verb_wayyiqtol=0, verb_qatal=0, verb_yiqtol=0,
        verb_participle=0, verb_inf_abs=0, verb_inf_con=0,
        neg_lo=0, neg_ein=0, yesh=0,
        total_words=0, total_verbs=0,
    )

    for w in word_nodes:
        c['total_words'] += 1
        lex = F.lex.v(w)
        sp  = F.sp.v(w)
        vt  = F.vt.v(w)

        # Independent function-word features (high frequency, low noise)
        if lex == LEX_ANOCHI:  c['pronoun_anochi'] += 1
        elif lex == LEX_ANI:   c['pronoun_ani'] += 1

        if lex == LEX_ASHER:   c['rel_asher'] += 1
        elif lex == LEX_SHE:   c['rel_she'] += 1

        if lex == LEX_YESH:    c['yesh'] += 1
        if lex == LEX_EIN:     c['neg_ein'] += 1
        if lex == LEX_LO:      c['neg_lo'] += 1

        # Verb form features
        if sp == 'verb':
            c['total_verbs'] += 1
            if vt == VT_WAYYIQTOL: c['verb_wayyiqtol'] += 1
            elif vt == VT_QATAL:   c['verb_qatal'] += 1
            elif vt == VT_YIQTOL:  c['verb_yiqtol'] += 1
            elif vt in (VT_PTCA, VT_PTCP): c['verb_participle'] += 1
            elif vt == VT_INF_ABS: c['verb_inf_abs'] += 1
            elif vt == VT_INF_CON: c['verb_inf_con'] += 1

    return c


# ---------------------------------------------------------------------------
# Data download helper
# ---------------------------------------------------------------------------

def download_bhsa(target_dir=None):
    """
    Attempt to download BHSA data via the text-fabric GitHub backend.
    This requires: pip install text-fabric[github]
    """
    print("Attempting to download BHSA data via text-fabric GitHub backend...")
    print("(requires:  pip install 'text-fabric[github]')\n")
    try:
        from tf.app import use
        # hoist=False avoids injecting into globals; we just want the download.
        A = use('etcbc/bhsa', hoist=False, silent=False)
        if A is None:
            raise RuntimeError("use() returned None — download may have failed.")
        print("\nDownload successful.")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nManual alternative:")
        print("  git clone --depth 1 https://github.com/etcbc/bhsa.git")
        print("  python3 01_feature_extraction_etcbc.py --data-path ./bhsa/tf/c")
        return False


# ---------------------------------------------------------------------------
# Main extraction routine
# ---------------------------------------------------------------------------

def run_extraction(data_path, output_path):
    """
    Load BHSA via the Fabric API and extract features for all dated units.

    Parameters
    ----------
    data_path : str or Path
        Directory containing the BHSA .tf feature files (e.g. .../bhsa/tf/c).
    output_path : str or Path
        Path for the output CSV.
    """
    data_path = Path(data_path).expanduser()

    if not data_path.exists():
        print(f"ERROR: BHSA data not found at {data_path}")
        print("\nRun the download step first:")
        print("  python3 01_feature_extraction_etcbc.py --download")
        sys.exit(1)

    # Check that at minimum otype.tf is present
    if not (data_path / 'otype.tf').exists():
        print(f"ERROR: {data_path}/otype.tf not found.")
        print("The path should contain the .tf feature files (e.g. otype.tf, lex.tf, sp.tf …)")
        sys.exit(1)

    try:
        from tf.fabric import Fabric
    except ImportError:
        print("ERROR: text-fabric not installed.")
        print("  pip install text-fabric[github]")
        sys.exit(1)

    print(f"Loading BHSA data from: {data_path}")
    TF = Fabric(locations=str(data_path), silent=True)

    # Load only the features we actually use — much faster than loading everything
    FEATURES_NEEDED = 'otype oslots otext book chapter verse lex sp vt'
    api = TF.load(FEATURES_NEEDED, silent=True)
    if api is None:
        print("ERROR: TF.load() returned None.  The data directory may be incomplete.")
        sys.exit(1)

    F = api.F   # Node feature accessor  (F.lex.v(w), F.sp.v(w), ...)
    L = api.L   # Locality API           (L.d(book_node, 'word'))
    T = api.T   # Text/section API       (T.nodeFromSection(('Genesis',)))

    print("BHSA loaded.  Extracting features...\n")

    rows = []
    for (unit_name, book_names, chap_range, date, sigma, period) in DATED_BOOK_SPECS:
        all_word_nodes = []

        for bname in book_names:
            # Resolve the book node.  T.nodeFromSection returns a single node
            # or None if not found.
            result = T.nodeFromSection((bname,))
            if result is None:
                print(f"  WARNING: book '{bname}' not found — skipping.")
                continue

            # result may be a single node (int) or a tuple depending on TF version
            book_node = result if isinstance(result, int) else result[0]

            # Collect chapter nodes within the requested range
            ch_nodes = L.d(book_node, 'chapter')
            for ch_node in ch_nodes:
                ch_num = F.chapter.v(ch_node)
                if ch_num is None:
                    continue
                if chap_range is not None:
                    if isinstance(chap_range, list):
                        # list of (lo, hi) inclusive ranges — e.g. [(1,1),(8,12)]
                        if not any(lo <= ch_num <= hi for lo, hi in chap_range):
                            continue
                    else:
                        lo, hi = chap_range
                        if not (lo <= ch_num <= hi):
                            continue
                all_word_nodes.extend(L.d(ch_node, 'word'))

        if not all_word_nodes:
            print(f"  WARNING: no word nodes collected for '{unit_name}' — skipping.")
            continue

        feat = extract_features(all_word_nodes, F)
        n = feat['total_words']
        row = {
            'unit':       unit_name,
            'date_bce':   date,
            'date_sigma': sigma,
            'period':     period,
            'n_words':    n,
            **{k: v for k, v in feat.items() if k != 'total_words'},
        }
        # Add per-1000-word rates directly in this file for convenience
        per = 1000.0
        for col in ['verb_wayyiqtol', 'verb_qatal', 'verb_yiqtol',
                    'verb_participle', 'verb_inf_abs', 'neg_lo', 'neg_ein', 'yesh']:
            row[f'rate_{col}'] = feat[col] / n * per if n > 0 else 0.0

        # Binary pair fractions (newer form / total)
        def safe_frac(num, den):
            return num / den if den > 0 else float('nan')

        row['frac_ani']              = safe_frac(feat['pronoun_ani'],
                                                  feat['pronoun_anochi'] + feat['pronoun_ani'])
        row['frac_she']              = safe_frac(feat['rel_she'],
                                                  feat['rel_asher'] + feat['rel_she'])
        row['frac_non_wayyiqtol']    = safe_frac(feat['verb_qatal'] + feat['verb_yiqtol'] + feat['verb_participle'],
                                                  feat['total_verbs'])
        row['frac_ein']              = safe_frac(feat['neg_ein'],
                                                  feat['neg_lo'] + feat['neg_ein'])

        rows.append(row)
        print(f"  {unit_name:<16}  {n:>7,} words  "
              f"wayyiqtol={feat['verb_wayyiqtol']:4d}  "
              f"אנכי={feat['pronoun_anochi']:3d}  אני={feat['pronoun_ani']:3d}  "
              f"אשר={feat['rel_asher']:4d}  ש={feat['rel_she']:3d}")

    if not rows:
        print("ERROR: No units were processed.  Check that book names match BHSA.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.to_csv(str(output_path), index=False)
    print(f"\nFeatures saved to: {output_path}")
    print(f"Units extracted: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Extract morphosyntactic features from BHSA/ETCBC Hebrew Bible data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Download BHSA data (one-time, requires text-fabric[github]):
  python3 01_feature_extraction_etcbc.py --download

  # Extract features using default data path:
  python3 01_feature_extraction_etcbc.py

  # Extract features from a manually-cloned BHSA repo:
  python3 01_feature_extraction_etcbc.py --data-path ./bhsa/tf/c
""")
    parser.add_argument('--download', action='store_true',
                        help='Download BHSA data via text-fabric (requires text-fabric[github])')
    parser.add_argument('--data-path', default=str(DEFAULT_BHSA_PATH),
                        help=f'Path to BHSA tf/c directory (default: {DEFAULT_BHSA_PATH})')
    parser.add_argument('--output', default='features_by_book.csv',
                        help='Output CSV path (default: features_by_book.csv)')
    args = parser.parse_args()

    if args.download:
        ok = download_bhsa()
        if not ok:
            sys.exit(1)
        # After download, continue to extraction
        if not Path(args.data_path).expanduser().exists():
            print("\nData downloaded.  Now run:  python3 01_feature_extraction_etcbc.py")
            sys.exit(0)

    out = Path(__file__).parent / args.output
    run_extraction(args.data_path, out)


if __name__ == '__main__':
    main()
