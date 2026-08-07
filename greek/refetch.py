"""
Re-acquire the Greek corpus at full extent.

The original download resolved one TLG work number per manifest entry.  For
authors whose corpus is split across many work numbers -- every orator, every
Life of Plutarch, every dialogue of Lucian -- that retrieved a single speech or
a single Life and discarded the rest.  Six of the seventeen Classical Attic
texts came through under 5,000 tokens as a result, which is exactly the end of
the scale the model most needs.

This script works from local clones of the two source repositories and collects
every prose work belonging to each entry, using the existing TEI extraction and
quote-removal code in 02_preprocess.py.

    /tmp/perseus   PerseusDL/canonical-greekLit
    /tmp/first1k   OpenGreekAndLatin/First1KGreek
"""
import os, re, glob, json, sys, importlib.util, unicodedata
import lxml.etree as etree

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("pre", os.path.join(HERE, "02_preprocess.py"))
pre = importlib.util.module_from_spec(spec)
sys.argv = ["02_preprocess.py"]
spec.loader.exec_module(pre)

REPOS = ["/tmp/perseus/data", "/tmp/first1k/data"]
TEI = "{http://www.tei-c.org/ns/1.0}"

# Author TLG id, plus an optional predicate on the work title, for every entry
# whose corpus is spread over multiple work numbers.  Entries not listed here
# keep whatever the original download produced.
SPEC = {
    # ── Classical Attic: the orators are one file per speech ───────────────
    "lysias_orations":        ("tlg0540", None),
    "demosthenes_philippics": ("tlg0014", None),
    "aeschines_orations":     ("tlg0026", None),
    "andocides_orations":     ("tlg0027", None),
    "hyperides_orations":     ("tlg0030", None),
    "isocrates_panegyricus":  ("tlg0010", None),
    # ── Atticizing: one file per Life, per dialogue, per oration ───────────
    "plutarch_lives":         ("tlg0007", "lives"),
    "plutarch_moralia":       ("tlg0007", "moralia"),
    "lucian_true_history":    ("tlg0062", None),
    "aelius_aristides_sacred": ("tlg0284", None),
    "maximus_tyrius_orations": ("tlg0537", None),
    "dionysius_thucydides":   ("tlg0081", "critical"),
    "dionysius_roman_antiquities": ("tlg0081", "antiquities"),
    # ── others split across books ─────────────────────────────────────────
    "theophrastus_historia_plantarum": ("tlg0093", {"tlg001"}),
    "theophrastus_characters": ("tlg0093", {"tlg009"}),
    "philo_de_opificio":      ("tlg0018", "opificio"),
    "philo_de_vita_mosis":    ("tlg0018", "mosis"),
}

# Plutarch's Moralia occupy the high work numbers in Perseus; the Lives the low
# ones.  Dionysius' Roman Antiquities is tlg001; the critical essays follow.
# Entries whose original file was the wrong work entirely, so a shorter result
# is the correct one: the first download mapped Characters to tlg001 (Historia
# Plantarum) and Historia Plantarum to tlg006 (the Metaphysics fragment), which
# put the same text in the corpus twice under two ids.
FORCE = {"theophrastus_characters"}

PLUT_MORALIA_FROM = 85
DION_ANTIQ = {"tlg001"}


def work_files(author):
    """Every Greek TEI file for an author across both repositories."""
    out = []
    for base in REPOS:
        d = os.path.join(base, author)
        if not os.path.isdir(d): continue
        for f in sorted(glob.glob(os.path.join(d, "*", "*.xml"))):
            if re.search(r"-(grc|1st1K-grc)\d*\.xml$", f) or ".grc" in os.path.basename(f):
                out.append(f)
    # prefer one edition per work number
    best = {}
    for f in out:
        wk = os.path.basename(os.path.dirname(f))
        if wk not in best or len(f) < len(best[wk]): best[wk] = f
    return [best[k] for k in sorted(best)]


def title_of(path):
    try:
        t = etree.parse(path)
        for el in t.iter(f"{TEI}title"):
            if el.text and el.text.strip(): return el.text.strip().lower()
    except Exception:
        pass
    return ""


def wanted(path, author, pred):
    wk = os.path.basename(os.path.dirname(path))
    ttl = title_of(path)
    if pred is None: return True
    if isinstance(pred, set): return wk in pred
    if pred == "lives":
        if author != "tlg0007": return True
        n = int(re.sub(r"\D", "", wk) or 0)
        return n < PLUT_MORALIA_FROM
    if pred == "moralia":
        n = int(re.sub(r"\D", "", wk) or 0)
        return n >= PLUT_MORALIA_FROM
    if pred == "antiquities": return wk in DION_ANTIQ
    if pred == "critical":    return wk not in DION_ANTIQ
    return pred in ttl


def main():
    man = json.load(open(os.path.join(HERE, "corpus_manifest.json"), encoding="utf-8"))
    meta = {e["id"]: e for e in man}
    report = []
    for eid, (author, pred) in SPEC.items():
        e = meta.get(eid)
        if e is None:
            print(f"  !! {eid} not in manifest"); continue
        files = [f for f in work_files(author) if wanted(f, author, pred)]
        if not files:
            print(f"  !! {eid}: no files found for {author}"); continue
        sents, kept = [], 0
        for f in files:
            try:
                txt = pre.extract_text_from_xml(f)
            except Exception as ex:
                print(f"     skip {os.path.basename(f)}: {type(ex).__name__}"); continue
            if not txt or len(txt) < 200: continue
            txt, _ = pre.remove_quotes_lexical(txt, heavy=bool(e.get("quote_heavy")))
            s = pre.tokenize(txt)
            if s: sents.extend(s); kept += 1
        n_tok = sum(len(x) for x in sents)
        old = os.path.join(HERE, "processed", f"{eid}.txt")
        prev = sum(len(l.split()) for l in open(old, encoding="utf-8")) if os.path.exists(old) else 0
        if n_tok <= prev and eid not in FORCE:
            print(f"  {eid:<32} {prev:>7} -> {n_tok:>7}  ({kept}/{len(files)} works)  KEEPING OLD")
            report.append(dict(id=eid, before=prev, after=prev, works=0, used=False))
            continue
        with open(old, "w", encoding="utf-8") as fh:
            for s in sents: fh.write(" ".join(s) + "\n")
        print(f"  {eid:<32} {prev:>7} -> {n_tok:>7}  ({kept}/{len(files)} works)")
        report.append(dict(id=eid, before=prev, after=n_tok, works=kept, used=True))
    json.dump(report, open(os.path.join(HERE, "refetch_report.json"), "w"), indent=2)
    tot_b = sum(r["before"] for r in report); tot_a = sum(r["after"] for r in report)
    print(f"\n  {len(report)} entries | {tot_b:,} -> {tot_a:,} tokens "
          f"({tot_a/max(tot_b,1):.1f}x)")


if __name__ == "__main__":
    main()
