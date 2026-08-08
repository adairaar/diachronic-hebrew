"""
Where the files are.

Every script in this repository used to carry the absolute path of the machine
it was written on.  That made the analysis unrunnable by anyone else: a clone
failed on the first stage with a missing file, which is a worse failure than
disagreeing results because it looks like the reader's mistake rather than ours.

Paths are resolved here instead, from the location of this file, so the tree can
sit anywhere.  Two helpers cover almost every use:

    f("big_features_500.csv")   a Hebrew-side file, wherever it lives
    g("greek_genre.json")       a Greek-side file

`f` searches for an existing file and, when writing, routes by name: the large
extracted matrices belong in hebrew/data, everything else in hebrew/results.
That mirrors the layout the repository already used before the scripts were
ported, so nothing moved on disk.

An environment variable DH_ROOT overrides the inferred root, which is useful for
running against a copy without editing anything.
"""
import os

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("DH_ROOT") or os.path.dirname(_here)

HEBREW = os.path.join(ROOT, "hebrew")
DATA = os.path.join(HEBREW, "data")
RESULTS = os.path.join(HEBREW, "results")
GREEK = os.path.join(ROOT, "greek")
PAPER = os.path.join(ROOT, "methods_paper_v4")
FIGURES = os.path.join(PAPER, "figures")
TABLES = os.path.join(PAPER, "tables")

for _d in (DATA, RESULTS, FIGURES, TABLES):
    os.makedirs(_d, exist_ok=True)

# Names that belong with the extracted matrices rather than with the results.
_DATA_PREFIXES = ("big_features_", "target_chunks_", "ngram_features_",
                  "poem_chunks", "chunk_features_", "feature_matrix_",
                  "chunk_preds")


def _is_data(name):
    return any(name.startswith(p) for p in _DATA_PREFIXES)


def f(name):
    """Absolute path of a Hebrew-side file, existing or about to be written."""
    name = os.path.basename(name)
    for d in (DATA, RESULTS, HEBREW):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(DATA if _is_data(name) else RESULTS, name)


def g(name):
    """Absolute path of a Greek-side file."""
    name = os.path.basename(name)
    for d in (GREEK, os.path.join(GREEK, "results")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(GREEK, name)


def fig(name):
    return os.path.join(FIGURES, os.path.basename(name))


def tab(name):
    return os.path.join(TABLES, os.path.basename(name))


def script(name):
    """A sibling script, for the few places one is read as data."""
    return os.path.join(HEBREW, os.path.basename(name))
