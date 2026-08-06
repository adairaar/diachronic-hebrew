"""
Chunk-level feature extraction for the Greek corpus.

Mirrors the Hebrew design: ~500-word chunks at sentence boundaries, then a wide
feature set dominated by high-frequency morphology and function-word syntax
rather than rare lexical shibboleths.

Feature families:
  A. 27 hand-built morphosyntactic rates (optative, infinitive, participle,
     particle, negation, -ττ-/-σσ-, sentence length ...) reusing the regexes
     from 03_feature_extraction.py
  B. top-250 word-form rates on the de-accented token stream
  C. character 3-gram and 4-gram rates on the de-accented, space-joined text
  D. function-word bigram rates over a closed list

Greek has no POS tagging comparable to BHSA, so families B-D substitute for the
Hebrew morphological features.  They are all high-frequency by construction.
"""
import os, re, json, collections, unicodedata, argparse
import numpy as np, pandas as pd
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "gfe", os.path.join(HERE, "03_feature_extraction.py"))
gfe = importlib.util.module_from_spec(spec)
gfe.HERE = HERE
gfe.PROC_DIR = os.path.join(HERE, "processed")
gfe.FEAT_DIR = os.path.join(HERE, "features")
gfe.MANIFEST = os.path.join(HERE, "corpus_manifest.json")
spec.loader.exec_module(gfe)

STRIP = gfe.strip_diacritics
GREEK = re.compile(r"^[α-ωϊϋ]+$")

# closed-class items whose bigrams carry syntactic profile
FUNC = set("""και δε μεν γαρ ουν αν τε ει εαν οτι ως ινα αλλα η ου ουκ ουχ μη
ο η το οι αι τα του της των τω τη τους τας ο τον την
εν εις εκ εξ απο δια κατα μετα περι προ προς συν υπο υπερ επι παρα ανα
αυτος αυτη αυτο αυτου αυτης αυτων αυτω αυτον αυτους
ουτος αυτη τουτο ουτοι ταυτα τουτου τουτων εκεινος
τις τι τινος οστις ος η ον ων οι
γε δη περ τοι ποτε ηδη νυν ετι ουτε μηδε αλλ ωσπερ ουδε""".split())


def load_sentences(eid):
    p = os.path.join(HERE, "processed", f"{eid}.txt")
    if not os.path.exists(p): return []
    out = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            t = line.strip().split()
            if t: out.append(t)
    return out


def chunk_sentences(sents, target=500):
    """Group whole sentences into chunks of about `target` tokens."""
    out, buf, n = [], [], 0
    for s in sents:
        buf.append(s); n += len(s)
        if n >= target:
            out.append(buf); buf, n = [], 0
    if buf:
        if out and n < target * 0.5: out[-1].extend(buf)
        else: out.append(buf)
    return out


def chunk_feats(sents, top_forms, top3, top4, top_bg):
    """Feature dict for one chunk (a list of sentences)."""
    f = dict(gfe.extract_features(sents))          # family A
    toks = [STRIP(w) for s in sents for w in s]
    toks = [t for t in toks if GREEK.match(t)]
    n = max(len(toks), 1)
    f["n_tok"] = len(toks)

    c = collections.Counter(toks)                   # family B
    for w in top_forms:
        f[f"w_{w}"] = 1000.0 * c[w] / n

    txt = " " + " ".join(toks) + " "                # family C
    L = max(len(txt), 1)
    c3 = collections.Counter(txt[i:i + 3] for i in range(len(txt) - 2))
    for gm in top3:
        f["c3_" + gm.replace(" ", "_")] = 1000.0 * c3[gm] / L
    c4 = collections.Counter(txt[i:i + 4] for i in range(len(txt) - 3))
    for gm in top4:
        f["c4_" + gm.replace(" ", "_")] = 1000.0 * c4[gm] / L

    bg = collections.Counter()                      # family D
    for s in sents:
        st = [STRIP(w) for w in s]
        for a, b in zip(st, st[1:]):
            if a in FUNC and b in FUNC: bg[(a, b)] += 1
    tb = max(sum(bg.values()), 1)
    for a, b in top_bg:
        f[f"bg_{a}_{b}"] = 1000.0 * bg[(a, b)] / tb
    return f


def main(target, n_forms, n_3, n_4, n_bg):
    man = json.load(open(os.path.join(HERE, "corpus_manifest.json"), encoding="utf-8"))
    texts = {}
    for e in man:
        s = load_sentences(e["id"])
        if s: texts[e["id"]] = s
    print(f"{len(texts)}/{len(man)} texts loaded")

    # ---- vocabulary chosen on the WHOLE corpus (unsupervised, no date labels) ----
    cf, c3, c4, cb = collections.Counter(), collections.Counter(), \
                     collections.Counter(), collections.Counter()
    for s in texts.values():
        st = [STRIP(w) for sn in s for w in sn]
        st = [t for t in st if GREEK.match(t)]
        cf.update(st)
        t = " " + " ".join(st) + " "
        c3.update(t[i:i + 3] for i in range(len(t) - 2))
        c4.update(t[i:i + 4] for i in range(len(t) - 3))
        for sn in s:
            ss = [STRIP(w) for w in sn]
            for a, b in zip(ss, ss[1:]):
                if a in FUNC and b in FUNC: cb[(a, b)] += 1
    top_forms = [w for w, _ in cf.most_common(n_forms)]
    top3 = [g for g, _ in c3.most_common(n_3)]
    top4 = [g for g, _ in c4.most_common(n_4)]
    top_bg = [g for g, _ in cb.most_common(n_bg)]
    print(f"vocab: {len(top_forms)} forms, {len(top3)} 3-grams, "
          f"{len(top4)} 4-grams, {len(top_bg)} function bigrams")

    meta = {e["id"]: e for e in man}
    rows = []
    for eid, sents in texts.items():
        e = meta[eid]
        chs = chunk_sentences(sents, target)
        for i, ch in enumerate(chs):
            fe = chunk_feats(ch, top_forms, top3, top4, top_bg)
            rows.append(dict(chunk_id=f"{eid}_c{i:03d}", unit=eid,
                             date_ce=e["date_ce"], register=e["register"],
                             genre=e["genre"], author=e["author"], **fe))
        print(f"  {eid:<34} {sum(len(s) for s in sents):7d} tok -> {len(chs):4d} chunks",
              flush=True)
    D = pd.DataFrame(rows)
    out = os.path.join(HERE, f"greek_chunks_{target}.csv")
    D.to_csv(out, index=False)
    nf = len([c for c in D.columns if c not in
              {"chunk_id", "unit", "date_ce", "register", "genre", "author", "n_tok"}])
    print(f"\n{len(D)} chunks | {D.unit.nunique()} texts | {nf} features -> {out}")
    print(D.groupby("register").agg(texts=("unit", "nunique"), chunks=("unit", "size")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--forms", type=int, default=250)
    ap.add_argument("--g3", type=int, default=200)
    ap.add_argument("--g4", type=int, default=150)
    ap.add_argument("--bg", type=int, default=120)
    a = ap.parse_args()
    main(a.target, a.forms, a.g3, a.g4, a.bg)
