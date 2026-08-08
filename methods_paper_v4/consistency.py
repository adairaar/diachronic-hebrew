"""
Mechanical consistency checks over the manuscript.

Reading 35 pages for consistency does not work; the eye slides over exactly the
things that matter.  These are the checks that can be made to fail loudly.

  1. Voice          first person plural in a single-author paper
  2. Spelling       British forms in a paper written in American English
  3. Stale terms    vocabulary left over from a superseded version
  4. Hard numbers   quantities typed into the prose instead of read from a macro
  5. Cross-refs     \\ref targets that no label defines, and labels never cited
  6. Macro hygiene  macros used but undefined; signed macros beside direction words
  7. Duplicates     the same quantity quoted from two different macros
"""
import json, os, re, sys
from collections import defaultdict

MS = "/home/claude/ms/main.tex"
NUM = "/home/claude/ms/numbers.tex"
src = open(MS).read()
nums = open(NUM).read()


def expand(text, depth=0):
    """Splice in \input files, so labels defined in tables/ are visible."""
    if depth > 3:
        return text
    def sub(m):
        f = os.path.join("/home/claude/ms", m.group(1))
        if not f.endswith(".tex"):
            f += ".tex"
        return expand(open(f).read(), depth + 1) if os.path.exists(f) else ""
    return re.sub(r"\\input\{([^}]+)\}", sub, text)


full = expand(src)          # main + everything it pulls in, for labels/refs

# strip comments so they are not searched
body = re.sub(r"(?<!\\)%.*", "", src)

fails = 0


def report(title, hits, fmt=lambda h: h):
    global fails
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")
    if not hits:
        print("  clean")
        return
    fails += len(hits)
    for h in hits[:40]:
        print("  " + fmt(h))
    if len(hits) > 40:
        print(f"  ... and {len(hits) - 40} more")


def ctx(m, w=48):
    s = max(0, m.start() - w)
    return " ".join(body[s:m.end() + w].split())


# ── 1. voice ────────────────────────────────────────────────────────────
PLURAL = r"\b(we|our|ours|us)\b"
hits = [ctx(m) for m in re.finditer(PLURAL, body, re.I)
        if not re.match(r"\\", body[max(0, m.start() - 1):m.start() + 1])]
report("1.  FIRST PERSON PLURAL (single-author paper)", hits)

# ── 2. British spellings ────────────────────────────────────────────────
BRIT = {
    "colour": "color", "behaviour": "behavior", "favour": "favor",
    "neighbour": "neighbor", "labour": "labor", "centre": "center",
    "metre": "meter", "fibre": "fiber", "analyse": "analyze",
    "analysed": "analyzed", "analysing": "analyzing",
    "recognise": "recognize", "recognised": "recognized",
    "emphasise": "emphasize", "emphasised": "emphasized",
    "characterise": "characterize", "characterised": "characterized",
    "generalise": "generalize", "summarise": "summarize",
    "normalise": "normalize", "regularise": "regularize",
    "modelling": "modeling", "modelled": "modeled",
    "labelled": "labeled", "labelling": "labeling",
    "cancelled": "canceled", "travelled": "traveled",
    "defence": "defense", "offence": "offense", "licence": "license",
    "practise": "practice", "programme": "program",
    "judgement": "judgment", "acknowledgement": "acknowledgment",
    "towards": "toward", "amongst": "among", "whilst": "while",
    "learnt": "learned", "spelt": "spelled", "grey": "gray",
}
hits = []
for b, a in BRIT.items():
    for m in re.finditer(rf"\b{b}\b", body, re.I):
        seg = ctx(m, 34)
        if re.search(r"(Centre for Bible|Talstra|Institute|University)", seg):
            continue        # proper noun, not a spelling choice
        hits.append(f"{b} -> {a}   ...{seg}...")
report("2.  BRITISH SPELLINGS", hits)

# ── 3. stale vocabulary from superseded versions ────────────────────────
STALE = {
    # "split conformal" is legitimate: the method section explains why it does
    # not apply here.  Bare "conformal" describing this paper's own intervals
    # is not.
    r"(?<!split )\bconformal\b(?! prediction)": "intervals are jackknife+ now",
    r"\bchunk\b": "the manuscript says 'passage'",
    r"\bcentred\b|\bcentring\b": "American: centered / centering",
    r"calibrated date ranges": "old title language",
    r"\bmorphosyntax carries a diachronic signal\b":
        "superseded: the signal is lexical",
}
hits = []
for pat, why in STALE.items():
    for m in re.finditer(pat, body, re.I):
        hits.append(f"[{why}]  ...{ctx(m, 40)}...")
report("3.  STALE VOCABULARY", hits)

# ── 4. numbers typed into the prose ─────────────────────────────────────
# allow: years used as historical dates, section/figure numbers, percentages
# that are definitional, and small integers that are counts of things named
ALLOW = {
    "586", "332", "539", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "12", "17", "26", "27", "34", "39", "40", "55", "56", "66", "0", "50",
    "100", "95", "68", "90", "20", "15", "11", "16", "24", "25", "30", "32",
    "1000", "2021", "1974", "1982", "0.05", "0.20", "0.001", "0.01",
}
hits = []
for m in re.finditer(r"(?<![\w.\\{])(\d+(?:\.\d+)?)(?![\w.}])", body):
    tok = m.group(1)
    if tok in ALLOW:
        continue
    seg = ctx(m, 40)
    if re.search(r"(Deut|Lev|Gen|Exod|Num|Isa|Jer|Judg|chapters?|verses?|"
                 r"BCE|CE|section|Table|Fig|S\d)", seg, re.I):
        continue
    hits.append(f"{tok:>8}   ...{seg}...")
# Literals that are correct as literals, each with the reason it is exempt.
# Anything not on this list is a finding.
OK_LITERAL = {
    "84": "attribution accuracy quoted from cited prior work, not a result here",
    "18": "verse reference Ezra 4:8--6:18",
    "200": "passage-size sweep grid, a design specification",
    "300": "passage-size sweep grid",
    "400": "passage-size sweep grid / chunk-length band bound",
    "700": "passage-size sweep grid / chunk-length band bound",
    "500": "passage-size sweep grid",
    "80": "1-2*alpha arithmetic at nominal 90%",
    "36": "1-2*alpha arithmetic at nominal 68%",
}
hits = [h for h in hits if h.split()[0] not in OK_LITERAL]
report("4.  NUMBERS TYPED INTO THE PROSE (not from a macro)", hits)

# ── 5. cross-references ─────────────────────────────────────────────────
labels = set(re.findall(r"\\label\{([^}]+)\}", full))
refs = set(re.findall(r"\\ref\{([^}]+)\}", full))
report("5a. \\ref TO A LABEL THAT DOES NOT EXIST", sorted(refs - labels))
unused = sorted(l for l in labels - refs
                if l.startswith(("tab:", "fig:")))
report("5b. FLOATS DEFINED BUT NEVER REFERENCED", unused)

# ── 6. macro hygiene ────────────────────────────────────────────────────
defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", nums))
builtin = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", src))
used = set(re.findall(r"\\([A-Za-z]+)", body))
LATEX = set("""ldots exp min max log ln arg bar hat acute tilde vec dot
delta nu tau theta phi psi omega gamma epsilon zeta kappa iota
propto rightarrow leftarrow Rightarrow to mapsto in notin subset supset
bibliography bibliographystyle pagestyle parindent newlength newcolumntype
makeatletter makeatother global noalign savedwidth thickcline hrule
fancyhf fancyheadoffset fancyfootoffset headrulewidth footrule lfoot rfoot
figurename textheight textsuperscript textquotesingle tfrac
DisableLigatures i j oe ae aa o l ss newline thickhline top u
varepsilon vrule vskip
begin end section subsection paragraph textbf textit emph cite
ref label input includegraphics caption centering footnotesize small large
Large item itemize enumerate newpage clearpage newgeometry restoregeometry
linenumbers usepackage documentclass title author date maketitle bf rm sf tt
hspace vspace quad qquad times cdot approx leq geq neq pm sim rho alpha beta
lambda sigma mu eta chi text mathrm frac sqrt sum prod int left right big
Big bigg Bigg langle rangle lfloor rfloor lceil rceil ensuremath newcommand
toprule midrule bottomrule addlinespace multicolumn textwidth linewidth
tabular table figure longtable endfirsthead endhead endfoot topsep itemsep
parskip baselineskip textless textgreater textbackslash today thepage
pageref nameref href url texttt underline sout hline cline arraybackslash
raggedright centerline noindent par smallskip medskip bigskip footnote
label textsc scriptsize normalsize itshape bfseries mdseries upshape
setlength tabcolsep arrayrulewidth extrarowheight renewcommand providecommand
DeclareMathOperator operatorname mathbb mathcal mathbf boldsymbol
""".split())
missing = sorted(used - defined - builtin - LATEX)
report("6a. MACRO-LOOKING COMMANDS THAT ARE NOT DEFINED", missing)

signed = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{\\ensuremath\{[+-]", nums))
DIRW = r"(earlier|later|too early|too late|apart|before|after|ahead|behind)"
hits = []
for mname in signed:
    for m in re.finditer(r"(.{0,70})\\" + mname + r"\b(.{0,55})", body):
        # search the surrounding prose only; a macro NAME containing "Before"
        # or "After" is not a direction word in the sentence
        # strip every macro name from the context: only prose counts
        around = re.sub(r"\\[A-Za-z]+", " ", m.group(1) + " " + m.group(2))
        if re.search(DIRW, around, re.I):
            hits.append(f"\\{mname}: ...{' '.join(m.group(0).split())}...")
report("6b. SIGNED MACRO BESIDE A DIRECTION WORD", hits)

# ── 6c. unbalanced math delimiters, paragraph by paragraph ──────────────
hits = []
for i, para in enumerate(re.split(r"\n\s*\n", body)):
    if para.lstrip().startswith("\\begin") or "\\end{" in para:
        continue
    n = len(re.findall(r"(?<!\\)\$", para))
    if n % 2:
        hits.append(" ".join(para.split())[:110] + " ...")
report("6c. PARAGRAPH WITH AN ODD NUMBER OF $ (runaway math mode)", hits)

# ── 6d. American English, by pattern rather than by word list ───────────
PROSE = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", body)
PROSE = re.sub(r"\$[^$]*\$", " ", PROSE)
BRIT_PAT = (r"\b\w{4,}is(e|es|ed|ing|ation|ations)\b|"
            r"\b(artefact|sceptic\w*|licence|defence|offence|pretence|storey|"
            r"towards|amongst|whilst|learnt|burnt|spelt|dreamt|leapt|"
            r"programme|fulfil|skilful|instil|distil|modelling|labelling|"
            r"cancelled|counsellor|focussed|judgement|acknowledgement|"
            r"analyse\w*|catalyse|paralyse|practise|behaviour\w*|colour\w*|"
            r"rigour|vigour|honour\w*|favour\w*|neighbour\w*|endeavour\w*)\b")
OK_ISE = re.compile(r"^(wise|rise\w*|arise\w*|precise|concise|promise\w*|"
                    r"premise\w*|surprise\w*|comprise\w*|otherwise|likewise|"
                    r"noise|raise\w*|revise\w*|advise\w*|devise\w*|"
                    r"pairwise|stepwise|exercise|enterprise|disguise\w*|"
                    r"franchise|merchandise|supervise\w*|unsupervised|"
                    r"promising|expertise|paradise|compromise\w*)$", re.I)
hits = []
for m in re.finditer(BRIT_PAT, PROSE, re.I):
    w = m.group(0)
    if OK_ISE.match(w):
        continue
    seg = " ".join(PROSE[max(0, m.start() - 40):m.end() + 30].split())
    if re.search(r"(Centre for Bible|Talstra)", seg):
        continue          # proper noun
    hits.append(f"{w}   ...{seg}...")
report("6d. BRITISH FORMS BY PATTERN (prose only)", hits)

# ── 7. the same quantity from two macros ────────────────────────────────
vals = defaultdict(list)
for name, val in re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.+?)\}\n", nums):
    v = re.sub(r"\\ensuremath\{|\}", "", val).strip()
    if re.fullmatch(r"[+-]?\d+(\.\d+)?", v) and name in used:
        vals[v].append(name)
hits = [f"{v:>10}  quoted via {', '.join(sorted(n))}"
        for v, n in sorted(vals.items()) if len(n) > 1]
print(f"\n{'=' * 76}\n7.  ADVISORY: one value reachable through several macros\n"
      f"{'=' * 76}")
print("  Usually coincidence -- two unrelated quantities that happen to be equal.")
print("  Listed so a genuine duplication cannot hide among them.  Not a failure.")
for h in hits:
    print("  " + h)

print(f"\n{'=' * 76}\n{fails} item(s) requiring action\n{'=' * 76}")
sys.exit(1 if fails else 0)
