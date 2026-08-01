"""
01_download_corpus.py
=====================
Downloads Greek prose texts from open-access GitHub repositories.

Sources
-------
Primary  : First1KGreek  (OpenGreekAndLatin/First1KGreek on GitHub)
Secondary: Perseus        (PerseusDL/canonical-greekLit on GitHub)

Strategy
--------
1. If the manifest entry has a `direct_url` field, use that.
2. Otherwise, build a file-path index from the GitHub API (cached) and
   resolve each entry by its `first1k_hint` (e.g. "tlg0016.tlg001"):
     a. Try First1KGreek index.
     b. Fall back to Perseus index.
3. Save raw TEI XML to data/raw/<id>.xml
4. Record provenance in data/raw/download_log.json

All downloads are cached; re-running skips already-downloaded files
unless --force is passed.

Usage
-----
    python 01_download_corpus.py [--force] [--id herodotus_histories]
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

HERE     = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(HERE, "data", "raw")
MANIFEST = os.path.join(HERE, "corpus_manifest.json")
LOG_FILE = os.path.join(RAW_DIR, "download_log.json")

F1K_BASE = "https://raw.githubusercontent.com/OpenGreekAndLatin/First1KGreek/master"
PER_BASE = "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master"

F1K_API  = "https://api.github.com/repos/OpenGreekAndLatin/First1KGreek/git/trees/master?recursive=1"
PER_API  = "https://api.github.com/repos/PerseusDL/canonical-greekLit/git/trees/master?recursive=1"

# ---------------------------------------------------------------------------
# Known direct URLs for texts that need them (confirmed via repository probe)
# These augment (and override) any `direct_url` field in the manifest.
# ---------------------------------------------------------------------------
KNOWN_URLS: dict[str, str] = {
    # First1KGreek
    "philo_de_opificio"      : f"{F1K_BASE}/data/tlg0018/tlg001/tlg0018.tlg001.1st1K-grc1.xml",
    "philo_de_vita_mosis"    : f"{F1K_BASE}/data/tlg0018/tlg014/tlg0018.tlg014.1st1K-grc1.xml",
    "epictetus_discourses"   : f"{PER_BASE}/data/tlg0557/tlg001/tlg0557.tlg001.perseus-grc2.xml",
    "arrian_anabasis"        : f"{PER_BASE}/data/tlg0074/tlg001/tlg0074.tlg001.perseus-grc2.xml",
    "lucian_true_history"    : f"{PER_BASE}/data/tlg0062/tlg001/tlg0062.tlg001.perseus-grc2.xml",
    "plotinus_enneads"       : f"{F1K_BASE}/data/tlg2000/tlg001/tlg2000.tlg001.1st1K-grc1.xml",
    "porphyry_life_plotinus" : f"{F1K_BASE}/data/tlg2034/tlg003/tlg2034.tlg003.1st1K-grc1.xml",
    "iamblichus_pythagorean" : f"{F1K_BASE}/data/tlg2023/tlg001/tlg2023.tlg001.1st1K-grc1.xml",
    "eusebius_church_history": f"{F1K_BASE}/data/tlg2018/tlg002/tlg2018.tlg002.1st1K-grc1.xml",
    "maximus_tyrius_orations": f"{F1K_BASE}/data/tlg0537/tlg010/tlg0537.tlg010.1st1K-grc1.xml",
    "clement_stromateis"     : f"{F1K_BASE}/data/tlg0555/tlg002/tlg0555.tlg002.1st1K-grc1.xml",
    "galen_natural_faculties": f"{F1K_BASE}/data/tlg0057/tlg017/tlg0057.tlg017.1st1K-grc1.xml",
    "theophrastus_characters": f"{F1K_BASE}/data/tlg0093/tlg001/tlg0093.tlg001.1st1K-grc1.xml",
    "theophrastus_historia_plantarum": f"{F1K_BASE}/data/tlg0093/tlg006/tlg0093.tlg006.1st1K-grc1.xml",
    "strabo_geography"       : f"{PER_BASE}/data/tlg0099/tlg001/tlg0099.tlg001.perseus-grc2.xml",
    "plutarch_lives"         : f"{F1K_BASE}/data/tlg0007/tlg146/tlg0007.tlg146.1st1K-grc1.xml",
    "aeschines_orations"     : f"{F1K_BASE}/data/tlg0026/tlg004/tlg0026.tlg004.1st1K-grc1.xml",
    # Perseus
    "herodotus_histories"            : f"{PER_BASE}/data/tlg0016/tlg001/tlg0016.tlg001.perseus-grc2.xml",
    "thucydides_history"             : f"{PER_BASE}/data/tlg0003/tlg001/tlg0003.tlg001.perseus-grc2.xml",
    "lysias_orations"                : f"{PER_BASE}/data/tlg0540/tlg001/tlg0540.tlg001.perseus-grc2.xml",
    "xenophon_anabasis"              : f"{PER_BASE}/data/tlg0032/tlg006/tlg0032.tlg006.perseus-grc2.xml",
    "xenophon_hellenica"             : f"{PER_BASE}/data/tlg0032/tlg001/tlg0032.tlg001.perseus-grc2.xml",
    "plato_republic"                 : f"{PER_BASE}/data/tlg0059/tlg030/tlg0059.tlg030.perseus-grc2.xml",
    "plato_laws"                     : f"{PER_BASE}/data/tlg0059/tlg034/tlg0059.tlg034.perseus-grc2.xml",
    "isocrates_panegyricus"          : f"{PER_BASE}/data/tlg0010/tlg001/tlg0010.tlg001.perseus-grc2.xml",
    "demosthenes_philippics"         : f"{PER_BASE}/data/tlg0014/tlg001/tlg0014.tlg001.perseus-grc2.xml",
    "andocides_orations"             : f"{PER_BASE}/data/tlg0027/tlg001/tlg0027.tlg001.perseus-grc2.xml",
    "aristotle_nicomachean"          : f"{PER_BASE}/data/tlg0086/tlg010/tlg0086.tlg010.perseus-grc2.xml",
    "aristotle_politics"             : f"{PER_BASE}/data/tlg0086/tlg025/tlg0086.tlg025.perseus-grc2.xml",
    "diodorus_siculus"               : f"{PER_BASE}/data/tlg0060/tlg001/tlg0060.tlg001.perseus-grc4.xml",
    "dionysius_roman_antiquities"    : f"{PER_BASE}/data/tlg0081/tlg001/tlg0081.tlg001.perseus-grc2.xml",
    "dionysius_thucydides"           : f"{PER_BASE}/data/tlg0081/tlg009/tlg0081.tlg009.perseus-grc2.xml",
    "josephus_jewish_war"            : f"{PER_BASE}/data/tlg0526/tlg004/tlg0526.tlg004.perseus-grc2.xml",
    "josephus_antiquities"           : f"{PER_BASE}/data/tlg0526/tlg001/tlg0526.tlg001.perseus-grc2.xml",
    "plutarch_moralia"               : f"{PER_BASE}/data/tlg0007/tlg112/tlg0007.tlg112.perseus-grc2.xml",
    "dio_chrysostom_orations"        : f"{PER_BASE}/data/tlg0612/tlg001/tlg0612.tlg001.perseus-grc2.xml",
    "appian_roman_history"           : f"{PER_BASE}/data/tlg0551/tlg017/tlg0551.tlg017.perseus-grc2.xml",
    "pausanias_description"          : f"{PER_BASE}/data/tlg0525/tlg001/tlg0525.tlg001.perseus-grc2.xml",
    "aelius_aristides_sacred"        : f"{PER_BASE}/data/tlg0284/tlg001/tlg0284.tlg001.perseus-grc2.xml",
    "cassius_dio_roman"              : f"{PER_BASE}/data/tlg0385/tlg001/tlg0385.tlg001.perseus-grc2.xml",
    # LXX texts (First1KGreek, tlg0527)
    "lxx_genesis"        : f"{F1K_BASE}/data/tlg0527/tlg001/tlg0527.tlg001.1st1K-grc1.xml",
    "lxx_exodus"         : f"{F1K_BASE}/data/tlg0527/tlg002/tlg0527.tlg002.1st1K-grc1.xml",
    "lxx_1kingdoms"      : f"{F1K_BASE}/data/tlg0527/tlg011/tlg0527.tlg011.1st1K-grc1.xml",
    "lxx_judith"         : f"{F1K_BASE}/data/tlg0527/tlg020/tlg0527.tlg020.1st1K-grc1.xml",
    "lxx_1maccabees"     : f"{F1K_BASE}/data/tlg0527/tlg023/tlg0527.tlg023.1st1K-grc1.xml",
    "lxx_2maccabees"     : f"{F1K_BASE}/data/tlg0527/tlg024/tlg0527.tlg024.1st1K-grc1.xml",
    # Jewish pseudepigrapha
    "testament_abraham"  : f"{F1K_BASE}/data/tlg1701/tlg001/tlg1701.tlg001.1st1K-grc1.xml",
    # Novels and vernacular Koine
    "vita_aesopi"        : f"{F1K_BASE}/data/tlg1765/tlg003/tlg1765.tlg003.1st1K-grc1.xml",
    "chariton_callirhoe" : f"{F1K_BASE}/data/tlg0693/tlg001/tlg0693.tlg001.1st1K-grc1.xml",
    "achilles_tatius"    : f"{PER_BASE}/data/tlg0532/tlg001/tlg0532.tlg001.perseus-grc2.xml",
    "ignatius_letters"   : f"{F1K_BASE}/data/tlg1443/tlg001/tlg1443.tlg001.1st1K-grc1.xml",
    # Late-BCE gap filler
    "apollodorus_library": f"{PER_BASE}/data/tlg0548/tlg001/tlg0548.tlg001.perseus-grc2.xml",
    # Late-Classical training addition
    "aeneas_tacticus"    : f"{PER_BASE}/data/tlg0058/tlg001/tlg0058.tlg001.perseus-grc2.xml",
    # New training texts (register-calibrated pipeline additions)
    "hyperides_orations"            : f"{PER_BASE}/data/tlg0030/tlg001/tlg0030.tlg001.perseus-grc2.xml",
    "philostratus_lives_sophists"   : f"{PER_BASE}/data/tlg0638/tlg001/tlg0638.tlg001.perseus-grc2.xml",
    "aelian_various_history"        : f"{PER_BASE}/data/tlg0545/tlg002/tlg0545.tlg002.perseus-grc2.xml",
    "libanius_orations"             : f"{F1K_BASE}/data/tlg2200/tlg001/tlg2200.tlg001.1st1K-grc1.xml",
    "john_gospel"                   : f"{PER_BASE}/data/tlg0031/tlg004/tlg0031.tlg004.perseus-grc2.xml",
    # Holdouts
    "polybius_histories"       : f"{PER_BASE}/data/tlg0543/tlg001/tlg0543.tlg001.perseus-grc2.xml",
    "mark_gospel"              : f"{PER_BASE}/data/tlg0031/tlg002/tlg0031.tlg002.perseus-grc2.xml",
    "matthew_gospel"           : f"{PER_BASE}/data/tlg0031/tlg001/tlg0031.tlg001.perseus-grc2.xml",
    "luke_gospel"              : f"{PER_BASE}/data/tlg0031/tlg003/tlg0031.tlg003.perseus-grc2.xml",
    "diogenes_laertius_lives"  : f"{PER_BASE}/data/tlg0004/tlg001/tlg0004.tlg001.perseus-grc2.xml",
}

# ---------------------------------------------------------------------------
# HTTP fetch
# ---------------------------------------------------------------------------

def fetch_url(url: str, retries: int = 3, delay: float = 2.0) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "diachronic-greek-pipeline/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise FileNotFoundError(f"404 Not Found: {url}")
            if attempt < retries - 1:
                print(f"    HTTP {e.code} on attempt {attempt+1}, retrying in {delay}s …")
                time.sleep(delay)
                delay *= 2
            else:
                raise
        except Exception as exc:
            if attempt < retries - 1:
                print(f"    Error ({exc}) on attempt {attempt+1}, retrying in {delay}s …")
                time.sleep(delay)
                delay *= 2
            else:
                raise


# ---------------------------------------------------------------------------
# Index builder (used only for entries with no known/direct URL)
# ---------------------------------------------------------------------------

def build_file_index(api_url: str, raw_base: str, cache_path: str) -> dict:
    """
    Build a dict: key 'tlgNNNN.tlgNNN' → list of raw file URLs.
    Cached to disk after first fetch.
    """
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    print(f"    Fetching file index from {api_url.split('/repos/')[1].split('/git')[0]} …")
    data = fetch_url(api_url)
    tree = json.loads(data)["tree"]

    index: dict[str, list[str]] = {}
    for item in tree:
        path = item.get("path", "")
        if not path.endswith(".xml"):
            continue
        m = re.search(r"(tlg\d+)[/\\](tlg\d+)[/\\]", path, re.IGNORECASE)
        if not m:
            continue
        key = f"{m.group(1).lower()}.{m.group(2).lower()}"
        raw_url = raw_base.rstrip("/") + "/master/" + path
        index.setdefault(key, []).append(raw_url)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"    Index saved: {len(index)} entries")
    return index


def prefer_greek(urls: list[str]) -> str | None:
    """Prefer a URL tagged 'grc'; avoid Latin/English translations."""
    grc = [u for u in urls if "grc" in u.lower()]
    if grc:
        return grc[0]
    non_lat = [u for u in urls if "lat" not in u.lower() and "eng" not in u.lower()]
    return non_lat[0] if non_lat else (urls[0] if urls else None)


# ---------------------------------------------------------------------------
# Download one entry
# ---------------------------------------------------------------------------

def download_entry(
    entry: dict,
    f1k_index: dict,
    per_index: dict,
    force: bool = False,
) -> dict:
    eid      = entry["id"]
    out_path = os.path.join(RAW_DIR, f"{eid}.xml")

    if os.path.exists(out_path) and not force:
        size = os.path.getsize(out_path)
        print(f"  [SKIP] {eid:45s} (cached, {size:,} bytes)")
        return {"id": eid, "status": "cached", "path": out_path}

    # Resolve URL: hardcoded table → manifest direct_url → index search
    url    = KNOWN_URLS.get(eid) or entry.get("direct_url")
    source = "known_url" if eid in KNOWN_URLS else ("direct_url" if url else None)

    if url is None:
        hint = entry.get("first1k_hint", "").lower()
        if hint in f1k_index:
            url    = prefer_greek(f1k_index[hint])
            source = "First1KGreek"
        elif hint in per_index:
            url    = prefer_greek(per_index[hint])
            source = "Perseus"

    if url is None:
        print(f"  [MISS] {eid:45s} — no URL resolved (hint={entry.get('first1k_hint')})")
        return {"id": eid, "status": "not_found", "path": None}

    try:
        print(f"  [DL]   {eid:45s} ← {source}")
        data = fetch_url(url)
        os.makedirs(RAW_DIR, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"         {len(data):,} bytes saved")
        return {"id": eid, "status": "ok", "path": out_path,
                "source": source, "url": url, "bytes": len(data)}
    except FileNotFoundError as exc:
        print(f"  [404]  {eid}: {exc}")
        return {"id": eid, "status": "404", "path": None, "url": url}
    except Exception as exc:
        print(f"  [ERR]  {eid}: {exc}")
        return {"id": eid, "status": "error", "path": None, "url": url, "error": str(exc)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--id", default=None, help="Download only this corpus ID.")
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)

    with open(MANIFEST, encoding="utf-8") as f:
        corpus = json.load(f)

    if args.id:
        corpus = [c for c in corpus if c["id"] == args.id]
        if not corpus:
            print(f"ID '{args.id}' not found in manifest.")
            sys.exit(1)

    # Build fallback indexes (cached; only used if known_url lookup fails)
    f1k_cache = os.path.join(RAW_DIR, "_f1k_index.json")
    per_cache = os.path.join(RAW_DIR, "_per_index.json")
    f1k_index, per_index = {}, {}
    entries_needing_index = [e for e in corpus
                             if e["id"] not in KNOWN_URLS and not e.get("direct_url")]
    if entries_needing_index:
        print("Building file indexes for unresolved entries …")
        f1k_index = build_file_index(F1K_API, F1K_BASE, f1k_cache)
        per_index = build_file_index(PER_API, PER_BASE, per_cache)

    print(f"\nDownloading {len(corpus)} texts …\n")
    log = []
    for entry in corpus:
        record = download_entry(entry, f1k_index, per_index, force=args.force)
        log.append(record)
        time.sleep(0.25)   # polite to GitHub

    # Save log
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    ok      = [r for r in log if r["status"] in ("ok", "cached")]
    missing = [r for r in log if r["status"] not in ("ok", "cached")]

    print(f"\n{'='*60}")
    print(f"Downloaded/cached : {len(ok)}")
    print(f"Missing/failed    : {len(missing)}")
    if missing:
        print("\nMissing entries:")
        for r in missing:
            print(f"  {r['id']:50s}  status={r['status']}")
    print(f"\nLog → {LOG_FILE}")


if __name__ == "__main__":
    main()
