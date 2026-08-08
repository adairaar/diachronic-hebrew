"""
Convert the figures to PLOS submission format and check them against the spec.

PLOS ONE wants TIFF or EPS, 300--600 DPI, width between 2.63 and 7.5 inches,
height at most 8.75 inches, RGB or grayscale, LZW-compressed.  A figure that
violates any of those is rejected at the format check before an editor sees it,
so each one is measured here rather than assumed.

Figures are rendered from the PDF sources at 600 DPI via pdftoppm, which
rasterises from the vector original rather than upsampling a bitmap.  The PDFs
remain the build's inputs; these files exist only for the submission system.

Numbering follows the order the figures are first referenced in the manuscript,
which is what the submission system expects, and is derived from main.tex rather
than hard-coded so it cannot fall out of step with the text.
"""
import os, re, subprocess, sys
from PIL import Image

MS = "/home/claude/ms"
SRC = f"{MS}/figures"
OUT = f"{MS}/submission"
DPI = 600
MIN_W_IN, MAX_W_IN, MAX_H_IN = 2.63, 7.5, 8.75

os.makedirs(OUT, exist_ok=True)
tex = open(f"{MS}/main.tex").read()

# figures defined after the Supporting Information heading are SI figures and
# take S-numbering, not a main-text figure number
si_at = tex.index("\\section*{Supporting Information}")

# order of first reference in the text
order, seen = [], set()
for m in re.finditer(r"\\ref\{(fig:[A-Za-z0-9_]+)\}", tex):
    if m.group(1) not in seen:
        seen.add(m.group(1)); order.append(m.group(1))

# map label -> source pdf, by reading each figure environment
label_to_pdf, is_si = {}, {}
for m in re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", tex, re.S):
    env = m.group(0)
    g = re.search(r"includegraphics\[[^\]]*\]\{figures/([^}]+)\}", env)
    l = re.search(r"\\label\{(fig:[A-Za-z0-9_]+)\}", env)
    if g and l:
        label_to_pdf[l.group(1)] = g.group(1)
        is_si[l.group(1)] = m.start() > si_at

# main-text figures keep reference order; SI figures are numbered separately
order = [l for l in order if not is_si.get(l)] + \
        [l for l in order if is_si.get(l)]

missing = [l for l in order if l not in label_to_pdf]
if missing:
    sys.exit(f"referenced but no figure environment found: {missing}")

print(f"{'file':<26}{'label':<18}{'pixels':>14}{'inches':>14}  status")
print("-" * 78)
fail = 0
n_main = sum(1 for l in order if not is_si.get(l))
for i, label in enumerate(order, 1):
    name = f"Fig{i}" if not is_si.get(label) else f"S{i - n_main}_Fig"
    pdf = os.path.join(SRC, label_to_pdf[label])
    if not os.path.exists(pdf):
        print(f"  MISSING SOURCE: {pdf}"); fail += 1; continue
    stem = f"{OUT}/{name}"
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-singlefile", pdf, stem],
                   check=True)
    im = Image.open(f"{stem}.png").convert("RGB")
    w_in, h_in = im.width / DPI, im.height / DPI

    # scale down if wider than the column limit; never scale up
    if w_in > MAX_W_IN:
        s = MAX_W_IN / w_in
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
        w_in, h_in = im.width / DPI, im.height / DPI
    im.save(f"{stem}.tif", compression="tiff_lzw", dpi=(DPI, DPI))
    os.remove(f"{stem}.png")

    bad = []
    if not (MIN_W_IN <= w_in <= MAX_W_IN): bad.append("width")
    if h_in > MAX_H_IN: bad.append("height")
    mb = os.path.getsize(f"{stem}.tif") / 1e6
    if mb > 10: bad.append(f"size {mb:.0f}MB")
    status = "ok" if not bad else "FAILS: " + ", ".join(bad)
    if bad: fail += 1
    print(f"  {name + '.tif':<24}{label:<18}{im.width}x{im.height:<7}"
          f"{w_in:>6.2f}x{h_in:<7.2f}{status}")

print("-" * 78)
print(f"{len(order)} figures -> {OUT}/  at {DPI} DPI, LZW TIFF, RGB")
print(f"limits: width {MIN_W_IN}--{MAX_W_IN} in, height <= {MAX_H_IN} in")
sys.exit(1 if fail else 0)
