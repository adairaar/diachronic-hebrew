"""Rebuild main.tex from the surviving sections plus the new material."""
import re, sys, pathlib

P = pathlib.Path("/home/claude/paper")
src = (P / "main.tex").read_text().split("\n")

def find(pat, start=0, required=True):
    rx = re.compile(pat)
    for i in range(start, len(src)):
        if rx.match(src[i].strip()):
            return i
    if required:
        sys.exit(f"MARKER NOT FOUND: {pat}")
    return None

# ── locate every boundary ────────────────────────────────────────────────────
i_abs      = find(r"\\section\*\{Abstract\}")
i_absend   = find(r"\\clearpage", i_abs)                 # abstract block ends here
i_corpus   = find(r"\\subsection\*\{Training corpus\}")
i_featex   = find(r"\\subsection\*\{Feature extraction\}")
i_tikh     = find(r"\\subsection\*\{Tikhonov")
i_post     = find(r"\\subsection\*\{Posterior computation\}")
i_hbres    = find(r"\\subsection\*\{Pentateuchal source dating")
i_prior    = find(r"\\section\*\{Prior sensitivity analysis\}")
i_arch     = find(r"\\section\*\{Archaism detection")
i_valid    = find(r"\\section\*\{Validation\}")
i_disc     = find(r"\\section\*\{Discussion\}")
i_concl    = find(r"\\section\*\{Conclusion\}")
i_supp     = find(r"\\section\*\{Supporting information\}")

for nm, a, b in [("corpus", i_corpus, i_featex), ("tikhonov", i_tikh, i_post),
                 ("hbvi-results", i_hbres, i_prior), ("archaism+genre", i_arch, i_valid),
                 ("validation", i_valid, i_disc), ("conclusion", i_concl, i_supp)]:
    assert a < b, f"{nm}: boundaries out of order"

# ── new material ─────────────────────────────────────────────────────────────
def blocks(path):
    txt = (P / path).read_text()
    out, cur, name = {}, [], None
    for ln in txt.split("\n"):
        m = re.match(r"%+ BLOCK: (\S+) %+", ln.strip())
        if m:
            if name: out[name] = "\n".join(cur).strip("\n")
            name, cur = m.group(1), []
        else:
            cur.append(ln)
    if name: out[name] = "\n".join(cur).strip("\n")
    return out

MB       = blocks("new_misc.tex")
MB.update(blocks("new_misc2.tex"))
abstract = (P / "new_abstract.tex").read_text().rstrip("\n")
corpus   = (P / "sec_training_corpus.tex").read_text().rstrip("\n")
tabcorp  = (P / "tab_corpus.tex").read_text().rstrip("\n")
tabtarg  = (P / "tab_targets.tex").read_text().rstrip("\n")
tableak  = (P / "tab_leakage.tex").read_text().rstrip("\n")
tabsub   = (P / "tab_subsources.tex").read_text().rstrip("\n")
validat  = (P / "new_validation.tex").read_text().rstrip("\n")
results  = (P / "new_results.tex").read_text().rstrip("\n")

# prior-sensitivity "Results" subsection is replaced wholesale
i_psres = find(r"\\subsection\*\{Results\}", i_prior)
assert i_psres < i_arch

# ── assemble ─────────────────────────────────────────────────────────────────
O = []
O += src[:i_abs]                                   # preamble + title block
O += [abstract, ""]
O += src[i_absend:i_corpus]                        # intro, background, data intro
O += [corpus, "", tabcorp, ""]
O += src[i_featex:i_tikh]                          # feature extraction, n-gram, OLS
O += [MB["tikhonov"], ""]
O += src[i_post:i_hbres]                           # posterior comp, HB-VI architecture
O += [MB["hbvi_results"], ""]
O += [src[i_prior]]                                # \section*{Prior sensitivity analysis}
O += ["", MB["prior_sensitivity_intro"], ""]
O += src[i_prior + 1:i_psres]                      # prior modes methodology
O += [MB["ps_results"], "", tabsub, ""]            # rewritten results + Delta table
O += [MB["lbh_score"], ""]                         # rescued from the cut section
#   <- archaism and genre-correction sections dropped entirely
O += [validat, "", tableak, "", results, "", tabtarg, ""]
O += src[i_disc:i_concl]                           # discussion + limitations
O += [MB["conclusion"], ""]
O += src[i_supp:]                                  # supporting info onward

# ── drop float environments tied to withdrawn claims ─────────────────────────
DEAD = ["fig3_oracle_jeremiah",      # holdout claim withdrawn
        "fig4_full_vs_resistant",    # resistant model withdrawn
        "fig5_genre_ratio",          # genre-correction section cut
        "fig6_genre_correction",
        "fig7_hbvi_comparison",      # head-to-head comparison cut
        "fig9_greek_validation",     # cross-language claim withdrawn
        "fig_loo_calibration",       # superseded by conformal
        "fig_s8_calibration",        # superseded by conformal
        "fig_s10_ci_vs_wordcount",   # interval widths superseded
        "fig_s11_key_posteriors"]    # posteriors superseded

def prune(lines, keys):
    out, i, killed = [], 0, []
    while i < len(lines):
        if re.match(r"\\begin\{(figure|table)\}", lines[i].strip()):
            j, env = i, re.match(r"\\begin\{(\w+)\}", lines[i].strip()).group(1)
            while j < len(lines) and not lines[j].strip().startswith(f"\\end{{{env}}}"):
                j += 1
            block = "\n".join(lines[i:j+1])
            hit = next((k for k in keys if k in block), None)
            if hit:
                killed.append(hit); i = j + 1; continue
        out.append(lines[i]); i += 1
    return out, killed

O = "\n".join(O).split("\n")      # blocks were appended as multi-line strings
O, killed = prune(O, DEAD)

# also drop the \paragraph* stubs that introduced those supplementary figures
STUB = {"fig3_oracle_jeremiah":"S13","fig_loo_calibration":"S13"}

out = "\n".join(O)

# repoint a caption cross-reference into the pruned comparison figure
out = out.replace(
    "overconfident in-sample; the \\hbvi{} model corrects this (Fig~\\ref{fig:hbvi_comparison}).",
    "overconfident in-sample.  This overconfidence is quantified under\n"
    "\\nameref{sec:validation} and is the reason the intervals reported in\n"
    "Table~\\ref{tab:targets} are conformal rather than parametric.")

(P / "main_new.tex").write_text(out)
print(f"  pruned {len(killed)} float environments: {', '.join(sorted(set(killed)))}\n")

# ── report ───────────────────────────────────────────────────────────────────
print(f"  original : {len(src):5d} lines")
print(f"  rebuilt  : {len(O):5d} lines   ({len(O)-len(src):+d})")
print(f"  dropped  : archaism section ({i_valid-i_arch} lines incl. genre correction)")
print(f"             old validation   ({i_disc-i_valid} lines)")

# dangling references to labels that no longer exist
labels = set(re.findall(r"\\label\{([^}]+)\}", out))
refs   = set(re.findall(r"\\(?:name)?ref\{([^}]+)\}", out))
missing = sorted(refs - labels)
print(f"\n  labels defined: {len(labels)}   referenced: {len(refs)}")
if missing:
    print("  DANGLING REFERENCES (must fix):")
    for m in missing:
        n = sum(1 for ln in O if f"ref{{{m}}}" in ln)
        print(f"    {m:<28} cited {n}x")
else:
    print("  no dangling references")
