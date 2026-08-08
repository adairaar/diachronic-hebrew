"""
Run the whole pipeline from the corpus, and diff the result against what is on
disk.

The point is not to produce results -- they already exist -- but to establish
that a reader who clones the repository and runs this gets the same numbers the
manuscript reports.  Two failures found by hand during this project (a feature
matrix that no longer matched the extractor that supposedly built it, and three
result files produced by ad-hoc commands that were never committed) are exactly
what this is meant to catch automatically from now on.

Stages run in dependency order.  Anything that reads BHSA comes first; every
later stage reads only files earlier stages wrote.  A stage that fails stops the
run, because a downstream result computed from a stale input is worse than no
result.

Usage:
    python3 run_pipeline.py --snapshot     record current outputs, then stop
    python3 run_pipeline.py                run everything and diff
    python3 run_pipeline.py --from 4       resume at stage 4
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import argparse, hashlib, json, os, subprocess, sys, time


SNAP = DH.f(".pipeline_snapshot.json")

# (label, command, [outputs it is responsible for])
STAGES = [
    ("extract anchored corpus",
     ["python3", "chunk_extract_big.py", "--target", "500", "--klex", "250"],
     ["big_features_500.csv", "big_features_500_ctyp.csv"]),
    ("extract undated targets",
     ["python3", "target_chunks.py", "--target", "500"],
     ["target_chunks_500.csv"]),
    ("extract n-gram variant",
     ["python3", "ngram_extract.py", "--targets", "500"],
     ["ngram_features_500.csv"]),
    ("corpus descriptive audit",
     ["python3", "corpus_audit.py"],
     ["chunk_sizes.json", "feature_counts.json"]),
    ("clause-type comparison",
     ["python3", "quicklobo.py"], []),
    ("clause-type leverage",
     ["python3", "clausetype_leverage.py"],
     ["clausetype_leverage.json"]),
    # the permutation count is passed explicitly: final_lobo.py defaults to 200
    # and the manuscript reports 300, so a clone must be told which to reproduce
    ("main leave-one-book-out",
     ["python3", "final_lobo.py", "300"],
     ["final_lobo_metrics.json", "final_lobo_books.csv"]),
    ("jackknife over books",
     ["python3", "jackknife.py"], ["jackknife.json", "jackknife.csv"]),

    ("nested configuration selection",
     ["python3", "nested_selection_var.py"], ["nested_selection_var.json"]),
    ("anchor sensitivity",
     ["python3", "anchor_sensitivity.py"], ["anchor_sensitivity.json"]),
    ("predict undated targets",
     ["python3", "predict_targets.py"], ["target_predictions_naive.csv"]),
    ("poems at three extractions",
     ["python3", "poems.py"], ["poem_predictions.csv"]),
    ("jackknife+ intervals",
     ["python3", "jackknife_plus.py"],
     ["jackknife_plus.json", "jackknife_plus_targets.csv",
      "jackknife_plus_residuals.csv"]),
    ("poem intervals to jackknife+",
     ["python3", "finalize_poems.py"], []),
    ("assemble published target table",
     ["python3", "finalize_targets.py"], ["target_predictions_final.csv"]),
    ("genre confound",
     ["python3", "genre_confound.py"], ["genre_confound.json"]),
    ("genre, symmetric correction",
     ["python3", "genre_symmetric.py"],
     ["genre_symmetric.json", "genre_symmetric_targets.csv"]),
    ("red-team ablations",
     ["python3", "red_team.py"], ["red_team.csv", "red_team.json"]),
    ("lexical diagnosis",
     ["python3", "lexical_diagnosis.py"], ["lexical_diagnosis.json"]),
    ("internal consistency",
     ["python3", "internal_consistency.py"],
     ["internal_consistency.csv", "chunk_preds.csv"]),
    ("block separations under genre screen",
     ["python3", "block_robustness.py"],
     ["block_robustness.csv", "block_robustness.json"]),
    ("dispersion under genre screen",
     ["python3", "disp_robustness.py"], ["disp_robustness.json"]),
    ("permutation stopping-point table",
     ["python3", "null_stability.py"],
     ["null_stability.csv", "null_stability.json"]),
    ("anchor-date bias direction",
     ["python3", "anchor_bias.py"], ["anchor_bias.json"]),
    ("feature scale test",
     ["python3", "feature_scale_test.py"], ["feature_scale_test.json"]),
    ("variationist share features",
     ["python3", "variationist_test.py"], ["variationist_test.json"]),
    ("share-encoding controls",
     ["python3", "share_control.py"], ["share_control.json"]),
    ("share pair provenance and subset sensitivity",
     ["python3", "share_provenance.py"], ["share_provenance.json"]),
    ("greek genre control",
     ["python3", os.path.join(DH.GREEK, "greek_genre.py")], ["greek/greek_genre.json"]),
]

# Outputs that are expected to differ run to run, with the reason.  Anything not
# listed here must reproduce byte for byte.
EXPECTED_DRIFT = {
    "chunk_preds.csv": "bootstrap RNG is seeded, but column order is not pinned",
}


def resolve(name):
    """An output's absolute path, wherever the layout puts it."""
    return DH.g(name.split("/", 1)[1]) if name.startswith("greek/") else DH.f(name)


def digest(path):
    p = resolve(path)
    if not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(65536), b""):
            h.update(blk)
    return h.hexdigest()


def all_outputs():
    return sorted({o for _, _, outs in STAGES for o in outs})


def snapshot():
    snap = {o: digest(o) for o in all_outputs()}
    json.dump(snap, open(SNAP, "w"), indent=2)
    have = sum(1 for v in snap.values() if v)
    print(f"snapshot: {have}/{len(snap)} outputs recorded -> {SNAP}")
    for o, v in sorted(snap.items()):
        if not v:
            print(f"  (absent) {o}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--from", dest="start", type=int, default=1)
    a = ap.parse_args()
    if a.snapshot:
        snapshot(); return

    if not os.path.exists(SNAP):
        sys.exit("no snapshot; run with --snapshot first")
    before = json.load(open(SNAP))

    t0 = time.time()
    for i, (label, cmd, outs) in enumerate(STAGES, 1):
        if i < a.start:
            print(f"[{i:>2}/{len(STAGES)}] {label:<40} skipped")
            continue
        t = time.time()
        print(f"[{i:>2}/{len(STAGES)}] {label:<40} ", end="", flush=True)
        log = os.path.join(DH.RESULTS, "pipeline_logs",
                           f"{i:02d}_{os.path.basename(cmd[1])}.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, cwd=DH.HEBREW, stdout=fh, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t
        if rc != 0:
            print(f"FAILED after {dt/60:.1f} min -- see {log}")
            sys.exit(f"stage {i} failed; downstream results would be stale")
        print(f"ok  {dt/60:>5.1f} min")

    print(f"\nall stages completed in {(time.time()-t0)/60:.0f} min\n")

    same, diff, missing = [], [], []
    for o in all_outputs():
        d = digest(o)
        if d is None:
            missing.append(o)
        elif before.get(o) is None:
            diff.append((o, "new"))
        elif d == before[o]:
            same.append(o)
        else:
            diff.append((o, "CHANGED"))

    print("=" * 74)
    print("REPRODUCIBILITY")
    print("=" * 74)
    print(f"  identical : {len(same)}")
    print(f"  changed   : {len([d for d in diff if d[1] == 'CHANGED'])}")
    print(f"  new       : {len([d for d in diff if d[1] == 'new'])}")
    print(f"  missing   : {len(missing)}")
    for o, why in sorted(diff):
        note = EXPECTED_DRIFT.get(o, "")
        print(f"    [{why:>7}] {o}" + (f"   ({note})" if note else ""))
    for o in missing:
        print(f"    [missing] {o}")
    unexplained = [o for o, why in diff
                   if why == "CHANGED" and o not in EXPECTED_DRIFT]
    print()
    if unexplained or missing:
        print(f"{len(unexplained) + len(missing)} output(s) do not reproduce.")
        sys.exit(1)
    print("Every output reproduces from the corpus.")


if __name__ == "__main__":
    main()
