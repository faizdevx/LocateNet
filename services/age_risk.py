"""Expected miss rate for an old reference photo, read from the age-gap evaluation results.

The numbers come from eval/results/age_gap.json (FG-NET, FNMR at the production threshold by
age-gap bucket, separately for photos taken before / after age 18). Nothing is hardcoded: if the
file is missing the app simply shows no estimate.
"""
import json
import os

BUCKETS = [(0, 2), (3, 5), (6, 10), (11, 20), (21, 200)]


def _bucket(years):
    for lo, hi in BUCKETS:
        if lo <= years <= hi:
            return f"{lo}-{hi}" if hi < 200 else f"{lo}+"
    return None


def load_age_results(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def expected_miss(results, years_old, age_at_photo=None):
    """Dict with bucket, fnmr, ci and the table used, or None when it cannot be estimated."""
    if not results or years_old is None:
        return None
    table_name = "all" if age_at_photo is None else ("child_enrolment" if age_at_photo < 18 else "adult_enrolment")
    table = results.get("fgnet_1to1", {}).get(table_name) or results.get("fgnet_1to1", {}).get("all", {})
    bucket = _bucket(int(round(max(0.0, years_old))))
    row = table.get(bucket) if bucket else None
    if row is None:
        table_name, row = "all", results.get("fgnet_1to1", {}).get("all", {}).get(bucket)
    if row is None:
        return None
    return {"bucket": bucket, "fnmr": row["fnmr"], "fnmr_ci": row["fnmr_ci"], "table": table_name}


def newest_photo_age(photos, now_age, years_old_fn):
    """(years_old, age_at_photo) of the most recent active reference photo whose age is known."""
    known = []
    for p in photos:
        if not p.get("active"):
            continue
        y = years_old_fn(p.get("photo_date"), p.get("age_at_photo"), now_age)
        if y is not None:
            known.append((y, p.get("age_at_photo")))
    return min(known, key=lambda t: t[0]) if known else (None, None)


def warning_text(years_old, estimate, threshold_years):
    if years_old is None or years_old <= threshold_years:
        return None
    text = f"Reference photo is ~{years_old:.0f} years old."
    if estimate:
        who = {"child_enrolment": "photos taken before age 18", "adult_enrolment": "adult photos"}.get(estimate["table"], "all photos")
        text += (f" Expected miss rate at this gap: ~{estimate['fnmr']:.0%} (FG-NET, {who}, {estimate['bucket']} years; "
                 "see docs/EVALUATION.md).")
    return text + " Consider adding a recent photo."


def default_results_path(base_dir):
    return os.path.join(base_dir, "eval", "results", "age_gap.json")
