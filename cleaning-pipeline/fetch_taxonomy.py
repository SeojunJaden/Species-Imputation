#!/usr/bin/env python3
"""
Fetch rank + ancestry for every taxon in the v2 dataset from the iNaturalist API.

Needed for Kellie's taxonomic cleanup rules (METHODOLOGY 5, spring 2026), which
turn on what rank a taxon is and what sits above it. iNat's own taxonomy is the
right authority here because the observations are iNat records.

Caches to RefreshedData/2026-09-14/taxonomy.json and only requests ids it does
not already hold, so re-running is cheap. Batches 30 ids per request at <=1 req/s,
the same courtesy limit pull_inat_api.py uses.

Usage:
  python cleaning-pipeline/fetch_taxonomy.py
"""

import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(ROOT, "RefreshedData", "2026-09-14")
FILTERED = os.path.join(V2, "filtered")
CACHE = os.path.join(V2, "taxonomy.json")

API = "https://api.inaturalist.org/v1/taxa/"
BATCH = 30
PAUSE = 1.0
UA = "Species-Imputation/1.0 (UC NRS reserve species imputation; research use)"


def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE, "w") as f:
        json.dump(cache, f)


def observed_taxon_ids():
    ids = set()
    for path in sorted(glob.glob(os.path.join(FILTERED, "*_Filtered.csv"))):
        df = pd.read_csv(path, usecols=["taxon_id"], low_memory=False)
        ids |= set(df["taxon_id"].dropna().astype(int).tolist())
    return ids


def fetch_batch(ids):
    url = API + ",".join(str(i) for i in ids)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r).get("results", [])
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            wait = 2 ** attempt
            print(f"    retry in {wait}s ({e})")
            time.sleep(wait)
    raise SystemExit("    giving up after 4 attempts")


def fetch_missing(ids, cache, label):
    todo = sorted(i for i in ids if str(i) not in cache)
    if not todo:
        print(f"[=] {label}: all {len(ids):,} already cached")
        return
    print(f"[*] {label}: fetching {len(todo):,} of {len(ids):,} "
          f"({-(-len(todo)//BATCH)} requests, ~{-(-len(todo)//BATCH)*PAUSE/60:.1f} min)")
    for n in range(0, len(todo), BATCH):
        chunk = todo[n:n + BATCH]
        for t in fetch_batch(chunk):
            cache[str(t["id"])] = dict(
                name=t.get("name"),
                rank=t.get("rank"),
                rank_level=t.get("rank_level"),
                parent_id=t.get("parent_id"),
                ancestor_ids=t.get("ancestor_ids", []),
                common_name=t.get("preferred_common_name"),
            )
        # Record ids the API returned nothing for, so we never re-request them.
        for i in chunk:
            cache.setdefault(str(i), None)
        done = min(n + BATCH, len(todo))
        if done % (BATCH * 25) == 0 or done == len(todo):
            save_cache(cache)
            print(f"    {done:,}/{len(todo):,}")
        time.sleep(PAUSE)
    save_cache(cache)


def main():
    cache = load_cache()
    print(f"[i] cache holds {len(cache):,} taxa")

    ids = observed_taxon_ids()
    fetch_missing(ids, cache, "observed taxa")

    # Second pass: parents of infraspecific taxa, needed to merge a subspecies
    # into its species when that species was never observed in our data.
    parents = set()
    for rec in cache.values():
        if rec and rec.get("rank") in ("subspecies", "variety", "form", "hybrid"):
            if rec.get("parent_id"):
                parents.add(int(rec["parent_id"]))
    fetch_missing(parents, cache, "parent taxa")

    resolved = {k: v for k, v in cache.items() if v}
    print(f"\n[+] {len(resolved):,} taxa resolved -> {os.path.relpath(CACHE, ROOT)}")
    ranks = pd.Series([v["rank"] for v in resolved.values()]).value_counts()
    print(ranks.head(15).to_string())


if __name__ == "__main__":
    main()
