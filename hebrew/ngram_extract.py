"""
N-gram feature extraction, appended to the 634-feature base.

  CHAR  character n-grams (3,4) over the consonantal skeleton
  POSTRI part-of-speech trigram rates
  LEXBI  lexeme bigram rates, restricted to the closed-class vocabulary
         (function words) where a bigram has a chance of recurring
  SYN    clause-relation trigrams and phrase-function bigrams

Vocabularies are chosen ONCE from the whole corpus by raw frequency, with no
reference to date, so vocabulary selection cannot leak the label.
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import os, re, json, argparse, collections
import numpy as np, pandas as pd
from tf.fabric import Fabric

TF_PATH = os.environ.get("BHSA_TF") or os.path.expanduser(
    "~/text-fabric-data/github/ETCBC/bhsa/tf/2021")

DIACRITICS = re.compile(r"[֑-ׇ]")

BHSA_NAME = {"Numbers":"Numeri","Deuteronomy":"Deuteronomium","Lamentations":"Threni",
             "Ezekiel":"Ezechiel","Obadiah":"Obadia","Jonah":"Jona","Malachi":"Maleachi",
             "Zephaniah":"Zephania","Habakkuk":"Habakuk","Micah":"Micha","Isaiah":"Jesaia",
             "Jeremiah":"Jeremia","Zechariah":"Sacharia","Ezra":"Esra","Nehemiah":"Nehemia",
             "1_Chronicles":"Chronica_I","2_Chronicles":"Chronica_II"}
RANGED = {
    "Isaiah_1":[("Jesaia",[(1,39)])], "Isaiah_2":[("Jesaia",[(40,55)])],
    "Isaiah_3":[("Jesaia",[(56,66)])], "Zechariah_1":[("Sacharia",[(1,8)])],
    "Zechariah_2":[("Sacharia",[(9,14)])],
    "Chronicles":[("Chronica_I",[(1,29)]),("Chronica_II",[(1,36)])],
    "Daniel":[("Daniel",[(1,1),(8,12)])],
    "Jer_oracle":[("Jeremia",[(1,6),(8,10),(12,16),(19,20),(22,23),(30,31),(46,51)])],
}
WHOLE = ["Amos","Hosea","Micah","Zephaniah","Nahum","Habakkuk","Lamentations",
         "Ezekiel","Obadiah","Joel","Jonah","Malachi","Haggai","Ezra","Nehemiah",
         "Esther","Ecclesiastes"]
CLOSED = {"prep","conj","art","nega","intj","advb","prin","prde","prps","inrg"}


def cons(w, F):
    return DIACRITICS.sub("", F.g_word_utf8.v(w) or "")


def char_ngrams(words, F, n):
    s = " " + " ".join(cons(w, F) for w in words) + " "
    return collections.Counter(s[i:i+n] for i in range(len(s)-n+1))


def main(targets, k_char3, k_char4, k_postri, k_lexbi):
    TF = Fabric(locations=[TF_PATH], silent="deep")
    api = TF.load("book chapter verse lex sp pdp g_word_utf8 function rela typ language",
                  silent="deep")
    F, L, T = api.F, api.L, api.T

    allw = [w for w in F.otype.s("word") if F.language.v(w) == "Hebrew"]
    print("building vocabularies (frequency only, date never consulted)...")
    c3 = char_ngrams(allw, F, 3); c4 = char_ngrams(allw, F, 4)
    V3 = [g for g, _ in c3.most_common(k_char3)]
    V4 = [g for g, _ in c4.most_common(k_char4)]
    sp_all = [F.sp.v(w) for w in allw]
    Vpt = [g for g, _ in collections.Counter(
        zip(sp_all, sp_all[1:], sp_all[2:])).most_common(k_postri)]
    lx = [(F.lex.v(w) if F.sp.v(w) in CLOSED else "·") for w in allw]
    Vlb = [g for g, _ in collections.Counter(zip(lx, lx[1:])).most_common(k_lexbi)
           if g != ("·", "·")]
    cl_all = list(F.otype.s("clause"))
    rl = [F.rela.v(c) for c in cl_all]
    Vrt = [g for g, _ in collections.Counter(zip(rl, rl[1:], rl[2:])).most_common(40)]
    ph_all = list(F.otype.s("phrase"))
    fn = [F.function.v(p) for p in ph_all]
    Vfb = [g for g, _ in collections.Counter(zip(fn, fn[1:])).most_common(60)]
    print(f"  char3 {len(V3)}  char4 {len(V4)}  postri {len(Vpt)}  "
          f"lexbi {len(Vlb)}  relatri {len(Vrt)}  funbi {len(Vfb)}")

    man = json.load(open(DH.f("corpus_manifest_v2.json")))
    units = []
    for grp in ("training", "holdouts"):
        for t in man[grp]:
            uid = t["id"]
            spec = RANGED.get(uid) or ([(BHSA_NAME.get(uid, uid), [(1, 999)])]
                                       if uid in WHOLE else None)
            if spec: units.append((uid, spec, t["date_bce"], t.get("genre")))

    for target in targets:
        rows = []
        for uid, spec, date, genre in units:
            words = []
            for book, ranges in spec:
                bn = T.nodeFromSection((book,))
                if bn is None: continue
                for ch in L.d(bn, "chapter"):
                    if any(s <= int(F.chapter.v(ch)) <= e for s, e in ranges):
                        words.extend(L.d(ch, "word"))
            if not words: continue
            verses, cur, curv = [], [], None
            for w in words:
                v = L.u(w, "verse"); v = v[0] if v else None
                if v != curv and cur: verses.append(cur); cur = []
                curv = v; cur.append(w)
            if cur: verses.append(cur)
            chunks, buf = [], []
            for v in verses:
                buf.extend(v)
                if len(buf) >= target: chunks.append(buf); buf = []
            if buf:
                if chunks and len(buf) < target * 0.5: chunks[-1].extend(buf)
                else: chunks.append(buf)

            for i, ch in enumerate(chunks):
                n = len(ch); f = {}
                g3 = char_ngrams(ch, F, 3); tot3 = max(sum(g3.values()), 1)
                for g in V3: f[f"c3_{g}"] = g3.get(g, 0) / tot3 * 1000
                g4 = char_ngrams(ch, F, 4); tot4 = max(sum(g4.values()), 1)
                for g in V4: f[f"c4_{g}"] = g4.get(g, 0) / tot4 * 1000
                sp = [F.sp.v(w) for w in ch]
                pt = collections.Counter(zip(sp, sp[1:], sp[2:]))
                for g in Vpt: f["pt_" + "|".join(g)] = pt.get(g, 0) / n * 1000
                lb_ = [(F.lex.v(w) if F.sp.v(w) in CLOSED else "·") for w in ch]
                lb = collections.Counter(zip(lb_, lb_[1:]))
                for g in Vlb: f["lb_" + "|".join(g)] = lb.get(g, 0) / n * 1000
                cls = sorted({c for w in ch for c in L.u(w, "clause")})
                rr = [F.rela.v(c) for c in cls]; ncl = max(len(cls), 1)
                rt = collections.Counter(zip(rr, rr[1:], rr[2:]))
                for g in Vrt: f["rt_" + "|".join(map(str, g))] = rt.get(g, 0) / ncl
                phs = sorted({p for w in ch for p in L.u(w, "phrase")})
                ff = [F.function.v(p) for p in phs]; nph = max(len(phs), 1)
                fb = collections.Counter(zip(ff, ff[1:]))
                for g in Vfb: f["fb_" + "|".join(map(str, g))] = fb.get(g, 0) / nph
                rows.append(dict(chunk_id=f"{uid}_c{i:03d}", unit=uid, date_bce=date,
                                 genre=genre, n_words=n, **f))
        D = pd.DataFrame(rows)
        out = DH.f(f"ngram_features_{target}.csv")
        D.to_csv(out, index=False)
        print(f"  ~{target}w: {len(D)} chunks x "
              f"{len([c for c in D.columns if c not in ('chunk_id','unit','date_bce','genre','n_words')])}"
              f" ngram features -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, nargs="+", default=[300, 500, 1000])
    ap.add_argument("--k_char3", type=int, default=200)
    ap.add_argument("--k_char4", type=int, default=150)
    ap.add_argument("--k_postri", type=int, default=120)
    ap.add_argument("--k_lexbi", type=int, default=120)
    a = ap.parse_args()
    main(a.targets, a.k_char3, a.k_char4, a.k_postri, a.k_lexbi)
