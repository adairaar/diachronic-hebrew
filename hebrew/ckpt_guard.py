"""
Refuse to resume a permutation checkpoint that was written against different
inputs.

Both null scripts checkpoint every few draws, because a run of several hundred
full-pipeline permutations will not survive an interrupted session.  Resuming is
therefore necessary.  Resuming from draws computed against a DIFFERENT feature
matrix is not, and it is silent: the run reports a p-value assembled partly from
a model that no longer exists.

That happened in this project.  A 1015-draw checkpoint survived a rebuild of the
corpus, and a later run asking for 300 draws skipped generation entirely and
reported the stale numbers.  Nothing failed; the p-value was simply wrong by an
amount nobody could see.

The fix is a fingerprint written beside the checkpoint, covering exactly what
determines the content of a draw: the feature matrix, and the RNG seed.  It
deliberately does NOT cover the script source.  A permutation costs minutes and
a full null costs hours, so invalidating one over an edited comment would push
the next person toward disabling the check -- which is how such guards die.
"""
import hashlib, os, shutil, sys


def fingerprint(paths, extra=""):
    h = hashlib.sha256()
    for p in sorted(paths):
        if os.path.exists(p):
            with open(p, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(blk)
        h.update(p.encode())
    h.update(extra.encode())
    return h.hexdigest()[:16]


def check(ckpt, inputs, extra=""):
    """Return True if an existing checkpoint may be resumed.

    Side effect: a checkpoint whose fingerprint does not match is renamed with a
    .stale suffix, so nothing is destroyed but nothing is silently reused."""
    fp_path = ckpt + ".fingerprint"
    now = fingerprint(inputs, extra)
    if not os.path.exists(ckpt):
        open(fp_path, "w").write(now)
        return False
    was = open(fp_path).read().strip() if os.path.exists(fp_path) else ""
    if was == now:
        return True
    n = 1
    while os.path.exists(f"{ckpt}.stale{n}"):
        n += 1
    shutil.move(ckpt, f"{ckpt}.stale{n}")
    print(f"  checkpoint fingerprint {was or '(absent)'} != {now}; "
          f"set aside as {os.path.basename(ckpt)}.stale{n} and starting fresh",
          file=sys.stderr, flush=True)
    open(fp_path, "w").write(now)
    return False
