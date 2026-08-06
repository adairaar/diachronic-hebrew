"""Back-matter cleanup after the HB-VI / prior-sensitivity cut."""
import pathlib, re, sys

p = pathlib.Path("/home/claude/paper/main_new.tex")
t = p.read_text(); n = 0

def rep(old, new, label):
    global t, n
    if old not in t: sys.exit(f"ANCHOR NOT FOUND (v3b): {label}")
    t = t.replace(old, new); n += 1

# ── prune float environments belonging to the cut sections ───────────────────
DEAD = ["fig8_prior_sensitivity", "fig_s9_elbo_convergence", "fig_circularity",
        "tab:s3_features", "tab:s4_robustness", "tab:s5_corpus", "tab:s6_prior_sweep"]
lines = t.split("\n"); out = []; i = 0; killed = []
while i < len(lines):
    m = re.match(r"\\begin\{(figure|table)\}", lines[i].strip())
    if m:
        env = m.group(1); j = i
        while j < len(lines) and not lines[j].strip().startswith(f"\\end{{{env}}}"):
            j += 1
        block = "\n".join(lines[i:j+1])
        hit = next((k for k in DEAD if k in block), None)
        if hit:
            killed.append(hit); i = j + 1; continue
    out.append(lines[i]); i += 1
t = "\n".join(out); n += len(killed)
print(f"  pruned {len(killed)} floats: {', '.join(sorted(set(killed)))}")

# ── rebuild the Supporting information list ──────────────────────────────────
si_start = t.find("\\section*{Supporting information}")
si_end   = t.find("\\section*{Data availability}")
if si_start < 0 or si_end < 0 or si_end <= si_start:
    sys.exit("SI section bounds not found")

SI = r"""\section*{Supporting information}

\paragraph*{S1 Appendix.}
\label{S1_Appendix}
\textbf{Feature definitions and BHSA extraction queries.}
Full specification of every feature, with the Text-Fabric query used to
compute it, so that the feature matrix can be regenerated from the BHSA
independently of the code in this repository.

\paragraph*{S2 Appendix.}
\label{S2_Appendix}
\textbf{Leave-one-out and permutation protocol.}
Pseudocode for the cross-validation loop, the in-fold screening step, the
conformal calibration, and the permutation null, together with the random
seeds used.

\paragraph*{S1 Table.}
\label{S1_Table}
\textbf{Screening power and false-discovery cost at $N = 23$.}
Retained feature counts, expected null counts, estimated false discovery
proportion, and minimum detectable $|\rho|$ at 80\% power, across nominal
$\alpha$.  See Table~\ref{tab:s2_power}.

\paragraph*{S2 Table.}
\label{S2_Table}
\textbf{Token counts for all units.}
Counts from the BHSA extraction used for every feature rate reported.
See Table~\ref{tab:s5_words}.

\paragraph*{S3 Table.}
\label{S3_Table}
\textbf{Specification curve.}
MAP ranges for every target across the seven model specifications of the
multiverse analysis, showing which targets are stable under respecification
and which are not.

\paragraph*{S1 Fig.}
\label{S1_Fig}
\textbf{Pairwise feature correlations.}
Correlation structure of the screened feature set, showing the redundancy
that motivates regularisation of the residual covariance.

"""

t = t[:si_start] + SI + t[si_end:]
n += 1

# ── remaining prose references ───────────────────────────────────────────────
rep("a ridge regression fitted alongside it.  The Prior Sensitivity Analysis section covers the three-mode\n"
    "sensitivity framework and the model-independent feature scoring used to\n"
    "make the inputs auditable.  The Validation section documents the prior-leakage",
    "a ridge regression fitted alongside it.  The Validation section documents the prior-leakage",
    "roadmap prior-sensitivity clause")

rep("$^a$Held out of the HB-VI model only.\n", "", "Table 1 HB-VI footnote")
rep("Habakkuk$^a$", "Habakkuk", "Table 1 Habakkuk marker")
rep("Daniel (Heb.)$^a$", "Daniel (Heb.)", "Table 1 Daniel marker")

rep(r"""($|\Delta_{AB}| < 30$~yr) are genuinely constrained by the linguistic""",
    r"""(a high data share in Table~\ref{tab:leakage}) are genuinely constrained by the linguistic""",
    "discussion delta (a)")
rep(r"""($|\Delta_{AB}| > 80$~yr), the circle is self-reinforcing: shift the""",
    r"""(a low data share), the circle is self-reinforcing: shift the""",
    "discussion delta (b)")

rep("All feature extraction scripts, Bayesian dating models, prior sensitivity",
    "All feature extraction scripts, dating models, cross-validation",
    "data availability sentence")

p.write_text(t)
print(f"v3b: {n} edits")
