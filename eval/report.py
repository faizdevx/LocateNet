"""`python -m eval report`: build docs/EVALUATION.md from eval/results/*.json (never by hand)."""
import json
import os

from eval.common import RESULTS_DIR, ROOT


def _load(name):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def pct(x, d=2):
    return "n/a" if x is None else f"{100 * float(x):.{d}f}%"


def ci(lo_hi, d=2):
    if not lo_hi or lo_hi[0] is None:
        return ""
    return f" [{pct(lo_hi[0], d)}, {pct(lo_hi[1], d)}]"


def num(x, d=3):
    return "n/a" if x is None else f"{float(x):.{d}f}"


def img(path, alt):
    return f"![{alt}](../{path})"


def headline(R):
    rows = []
    v, o, c, b, r, lat = R["verification_lfw"], R["openset_lfw"], R["calibration"], R["bias"], R["robustness"], R["latency"]
    if v:
        rows.append(("LFW 1:1 accuracy (10-fold), enrolled pairs / end to end",
                     f"{pct(v['accuracy_enrolled_pairs_mean'])} ± {pct(v['accuracy_enrolled_pairs_std'])} / "
                     f"{pct(v['accuracy_mean'])} ± {pct(v['accuracy_std'])}"))
    if o:
        g = o["splits"]["test"]["by_gallery_size"]["1000"]
        f = g["fnir_at_fpir"]["0.01"]
        rows.append(("FNIR @ FPIR=1%, gallery 1000 (test)", pct(f["fnir"]) + ci(f["fnir_ci"])))
        rows.append(("Rank-1 identification, gallery 1000 (test)", pct(g["rank1"]["rate"]) + ci(g["rank1"]["ci"])))
    if c:
        m = c["metrics_test"]
        rows.append(("ECE raw (min-max) -> calibrated (" + c["chosen"] + ")",
                     f"{num(m['raw_minmax']['ece'])} -> {num(m[c['chosen']]['ece'])}"))
    if b:
        d = b["disparity"]["fmr_diagnostic"]
        ratio = d.get("ratio") or d.get("ratio_nonzero_groups")
        lo_g = d["min_group"] if d.get("ratio") else d.get("min_nonzero_group")
        rci = d["ci"] if d.get("ratio") else d.get("ci_nonzero_groups")
        txt = "n/a" if ratio is None else f"{ratio:.2f}x ({d['max_group']} vs {lo_g})"
        if rci and rci[0] is not None:
            txt += f" [{rci[0]:.2f}, {rci[1]:.2f}]"
        rows.append(("Worst/best group FMR ratio at pooled FMR 1e-4" + (" (LFW fallback; Indian group too small to measure)"
                     if "FALLBACK" in b.get("data", "") else ""), txt))
    if r:
        w = r["minimum_usable_face_width"]["width_px"]
        rows.append(("Minimum usable face width", "n/a" if w is None else f"{w} px"))
    if lat:
        import config
        frames = lat.get("process_frame_by_det_size", {}).get(str(config.DET_SIZE[0]), lat["process_frame"])
        p = frames.get("10")
        if p:
            rows.append((f"p95 latency, 10-face photo (CPU, det_size {config.DET_SIZE[0]})",
                         f"{p['process_frame']['p95_ms']:.0f} ms"))
    out = ["| Metric | Value [95% CI] |", "|---|---|"] + [f"| {a} | {b_} |" for a, b_ in rows]
    return "\n".join(out)


def sec_embed(R):
    e = R["embed_lfw"]
    if not e:
        return ""
    f = e["fte"]
    return "\n".join([
        "## 1. Pipeline and data",
        "",
        "Every image goes through the production `FaceService` (InsightFace `buffalo_l`, det_size 640, "
        "`get_quality_embedding` with `DET_SCORE_MIN`) and the same L2 normalisation as `SearchService`. "
        "When an LFW image shows several faces, the largest face nearest the centre is used "
        "(faces at least half as large as the biggest one, then closest to the centre).",
        "",
        f"- LFW images: {f['n_images']}; enrolled: {f['ok']}; low quality (det_score < DET_SCORE_MIN): "
        f"{f['low_quality']}; no face found: {f['no_face']}.",
        f"- **Failure-to-enrol rate (FTE): {pct(f['fte_rate'], 3)}.** Failed images stay in the protocols as misses.",
        "",
    ])


def sec_verification(R):
    v = R["verification_lfw"]
    if not v:
        return ""
    t = v["tar_at_far"]
    return "\n".join([
        "## 2. 1:1 verification sanity check (LFW standard protocol)",
        "",
        "Protocol: official View 2 `pairs.txt` (10 folds x 600 pairs). For each fold the accuracy threshold is "
        "chosen on the other nine folds.",
        "",
        "| Metric | Value |", "|---|---|",
        f"| Accuracy, end to end (failed enrolment = reject) | {pct(v['accuracy_mean'])} ± {pct(v['accuracy_std'])} |",
        f"| Accuracy, pairs where both images enrolled (comparable to published) | "
        f"{pct(v['accuracy_enrolled_pairs_mean'])} ± {pct(v['accuracy_enrolled_pairs_std'])} |",
        f"| Pairs containing an image that failed to enrol | {v['n_pairs_with_failed_image']} of {v['n_pairs']} |",
        f"| TAR @ FAR=0.1% | {pct(t['0.001']['tar'])} (threshold {num(t['0.001']['threshold'])}) |",
        f"| TAR @ FAR=1% | {pct(t['0.01']['tar'])} (threshold {num(t['0.01']['threshold'])}) |",
        f"| Distinct images in pairs.txt that failed to enrol | {v['n_pair_images_failed_enrol']} |",
        "",
        img(v["plot"], "LFW ROC"),
        "",
        f"Interpretation: on pairs where both images enrolled, accuracy is {pct(v['accuracy_enrolled_pairs_mean'])}, in "
        "line with the published 99.7-99.8% for buffalo_l, so alignment and embedding are wired correctly. The "
        f"end-to-end figure ({pct(v['accuracy_mean'])}) is lower because the production detector (det_thresh 0.5, "
        "DET_SCORE_MIN) rejects some LFW images; published numbers use pre-aligned crops and never fail to enrol. "
        "TAR@FAR=0.1% rests on only 3 of 3000 impostor pairs and is noisy; the open-set results below are the ones "
        "that matter for the app.",
        "",
    ])


def sec_openset(R):
    o = R["openset_lfw"]
    if not o:
        return ""
    lines = ["## 3. Open-set 1:N identification (core experiment)", "", o["protocol"], "",
             "Test split (identities never used for any threshold choice):", "",
             "| Gallery | Mated / non-mated probes | FNIR @ FPIR 0.1% | FNIR @ FPIR 1% | FNIR @ FPIR 5% | Threshold @ 1% | Rank-1 | Rank-5 |",
             "|---|---|---|---|---|---|---|---|"]
    for size, g in o["splits"]["test"]["by_gallery_size"].items():
        f = g["fnir_at_fpir"]
        lines.append(
            f"| {g['gallery_size']} | {g['n_mated_probes']} / {g['n_nonmated_probes']} | "
            f"{pct(f['0.001']['fnir'])}{ci(f['0.001']['fnir_ci'])} | {pct(f['0.01']['fnir'])}{ci(f['0.01']['fnir_ci'])} | "
            f"{pct(f['0.05']['fnir'])}{ci(f['0.05']['fnir_ci'])} | {num(f['0.01']['threshold'])} "
            f"[{num(f['0.01']['threshold_ci'][0])}, {num(f['0.01']['threshold_ci'][1])}] | "
            f"{pct(g['rank1']['rate'])}{ci(g['rank1']['ci'])} | {pct(g['rank5']['rate'])} |")
    g = o["splits"]["test"]["by_gallery_size"]["1000"]
    old = g["at_threshold"].get("MATCH_THRESHOLD")
    small = o["splits"]["test"]["by_gallery_size"]["100"]["fnir_at_fpir"]["0.01"]["fnir"]
    big = o["splits"]["test"]["by_gallery_size"]["all"]["fnir_at_fpir"]["0.01"]["fnir"]
    lines += ["", img("eval/plots/openset_det.png", "Open-set DET"), "", img("eval/plots/score_hist.png", "Rank-1 score histogram"), ""]
    if old:
        lines.append(f"At the Week 1 hand-picked threshold ({num(old['threshold'], 2)}) and gallery 1000: FNIR "
                     f"{pct(old['fnir'])}, FPIR {pct(old['fpir'])}.")
        lines.append("")
    mb = g.get("miss_breakdown_at_fpir_0.01", {})
    sizes = o["splits"]["test"]["by_gallery_size"]
    t_small, t_big = sizes["100"]["fnir_at_fpir"]["0.01"]["threshold"], sizes["all"]["fnir_at_fpir"]["0.01"]["threshold"]
    f01 = g["fnir_at_fpir"]["0.001"]["fnir"]
    lines.append(f"Interpretation: a larger gallery gives impostors more chances to reach rank 1, so the threshold for "
                 f"FPIR 1% rises from {num(t_small)} (100 people) to {num(t_big)} ({sizes['all']['gallery_size']} people). "
                 f"On LFW the genuine scores sit far above it, so FNIR stays about the same ({pct(small)} vs {pct(big)}). "
                 "At gallery 1000 the misses split into: right person at rank 1 but below threshold "
                 f"{pct(mb.get('right_identity_below_threshold'))}, wrong person at rank 1 "
                 f"{pct(mb.get('wrong_identity_at_rank1'))}, probe failed to enrol {pct(mb.get('failed_to_enrol'))} "
                 "(each as a share of all mated searches).")
    lines.append("")
    lines.append(f"At FPIR 0.1% the picture changes: FNIR jumps to {pct(f01)} at gallery 1000, because that threshold is set "
                 "by the two or three highest non-mated scores, which are not ordinary impostors (see 'Label noise and "
                 "look-alikes' below). With about 1,900 non-mated searches, FPIR 0.1% is not a stable operating point on LFW.")
    lines.append("")
    return "\n".join(lines)


def sec_operating(R):
    op = R["operating_point"]
    if not op:
        return ""
    rec = op["recommended"]
    return "\n".join([
        "## 4. Operating point", "",
        f"Chosen on the tune split: **MATCH_THRESHOLD = {rec['MATCH_THRESHOLD']}** (FPIR {pct(rec['match_fpir'], 1)}), "
        f"**HIGH_CONF_THRESHOLD = {rec['HIGH_CONF_THRESHOLD']}** (FPIR {pct(rec['high_fpir'], 1)}). "
        "Full table, alert-load model and reasoning: [OPERATING_POINT.md](OPERATING_POINT.md).", ""])


def sec_calibration(R):
    c = R["calibration"]
    if not c:
        return ""
    m = c["metrics_test"]
    rows = "\n".join(f"| {k} | {num(v['ece'], 4)} | {num(v['brier'], 4)} |" for k, v in m.items())
    ex = ", ".join(f"{k} -> {pct(v, 0)}" for k, v in c["example_predictions"].items())
    return "\n".join([
        "## 5. Score calibration", "",
        f"Rank-1 cosine -> P(correct mated match), fitted on the tune split (gallery {c['gallery_size']}), "
        f"evaluated on the test split ({c['n_test_mated']} mated / {c['n_test_nonmated']} non-mated searches), "
        f"both re-weighted to a **{pct(c['prior'], 0)} mated prior**.", "",
        "| Method | ECE (test) | Brier (test) |", "|---|---|---|", rows, "",
        img(c["plot"], "Reliability diagram"), "",
        f"Chosen: **{c['chosen']}** ({c.get('selection_rule', '')}), exported as plain JSON to `{c['export']}`. "
        f"Example outputs: {ex}.", "",
        _middle(c), "",
        "**Base-rate caveat.** The probability is only valid when about "
        f"{pct(c['prior'], 0)} of searched faces really belong to an enrolled person. In a crowd photo most faces "
        "are strangers, so the true prior may be much lower, and the probability for the same score drops. "
        "To re-target prior pi' from the fitting prior pi: odds' = odds x (pi'/(1-pi')) / (pi/(1-pi)). The prior "
        "is `CALIBRATION_PRIOR` in `config.py`.", ""])


def _middle(c):
    m = c.get("middle_range_0.05_0.95")
    if not m or m.get("n", 0) == 0:
        return ""
    return (f"Interpretation: LFW scores are so well separated that only {pct(m['share_of_searches'])} of searches "
            f"(n={m['n']}) get a predicted probability between 5% and 95%, which is why every ECE is small. In that "
            f"middle range the calibrator predicts {pct(m['mean_predicted'], 0)} on average while "
            f"{pct(m['observed_correct'], 0)} were in fact correct, so it is conservative there. Raw min-max scores are "
            "badly calibrated (they spread non-mated searches over 0-0.3 and are over-confident).")


def _ratio_txt(x):
    if x.get("ratio") is not None:
        r = f"**{x['ratio']:.2f}x** ({x['max_group']} vs {x['min_group']})"
        if x.get("ci") and x["ci"][0] is not None:
            r += f", 95% CI {x['ci'][0]:.2f}-{x['ci'][1]:.2f}"
        return r
    r = f"unbounded ({x.get('min_group')} has zero errors)"
    if x.get("ratio_nonzero_groups"):
        r += (f"; among groups with errors: **{x['ratio_nonzero_groups']:.2f}x** ({x['max_group']} vs "
              f"{x['min_nonzero_group']})")
        c = x.get("ci_nonzero_groups")
        r += f", 95% CI {c[0]:.2f}-{c[1]:.2f}" if c and c[0] is not None else ", CI not estimable (errors too rare)"
    return r


def sec_bias(R):
    b = R["bias"]
    if not b:
        return ""
    fallback = "FALLBACK" in b.get("data", "")
    lines = ["## 6. Demographic bias audit", ""]
    if fallback:
        lines += ["> **Fallback, weaker evidence.** RFW was not available (it needs a licence request). Groups come "
                  "from the Kumar et al. LFW attribute classifier scores, which are noisy; LFW is mostly white and "
                  "male; some groups have very few identities. Treat these numbers as a smoke alarm, not a verdict.", "",
                  f"Group rule: {b['group_rule']}.", ""]
    proto = b.get("protocol", "")
    lines += [proto[proto.find("Views"):] if "Views" in proto else proto, "",
              f"Operating threshold {num(b['threshold'])} (from `{b['threshold_source']}`); diagnostic threshold "
              f"{num(b['diagnostic_threshold'])} (pooled impostor FMR = {b['diagnostic_fmr_target']:g}).", "",
              "| Group | Identities | Genuine / impostor pairs | FMR @ operating | FMR @ diagnostic | FNMR @ operating | Impostor p99.9 / max |",
              "|---|---|---|---|---|---|---|"]

    def zero(s, key):
        up = s.get(key + "_upper_if_zero")
        return f" (0 errors, <= {up:.1e})" if up else ""
    for g, s in b["race_1to1"].items():
        lines.append(f"| {g} | {s['n_identities']} | {s['n_genuine']} / {s['n_impostor']:,} | "
                     f"{s['fmr']:.1e}{zero(s, 'fmr')} | {s['fmr_diagnostic']:.1e} "
                     f"[{s['fmr_diagnostic_ci'][0]:.1e}, {s['fmr_diagnostic_ci'][1]:.1e}] | "
                     f"{pct(s['fnmr'])}{ci(s['fnmr_ci'])}{zero(s, 'fnmr')} | {num(s['impostor_p999'])} / {num(s['impostor_max'])} |")
    if b.get("gender_1to1"):
        lines += ["", "| Gender | Identities | FMR @ operating | FMR @ diagnostic | FNMR @ operating |", "|---|---|---|---|---|"]
        for g, s in b["gender_1to1"].items():
            lines.append(f"| {g} | {s['n_identities']} | {s['fmr']:.1e}{zero(s, 'fmr')} | {s['fmr_diagnostic']:.1e} | "
                         f"{pct(s['fnmr'])}{ci(s['fnmr_ci'])} |")
    if b.get("race_openset_gallery1000"):
        lines += ["", "Open-set (gallery 1000, test split), per probe group, at the operating threshold:", "",
                  "| Group | Identities | Mated / non-mated probes | FNIR | FPIR |", "|---|---|---|---|---|"]
        for g, s in b["race_openset_gallery1000"].items():
            lines.append(f"| {g} | {s['n_identities']} | {s['n_mated']} / {s['n_nonmated']} | {pct(s['fnir'])}{ci(s['fnir_ci'])} | "
                         f"{pct(s['fpir'])}{ci(s['fpir_ci'])} |")
    d = b["disparity"]
    lines += ["", "Disparity (max/min across race groups):", "",
              f"- FMR at the operating threshold: {_ratio_txt(d['fmr'])}.",
              f"- FMR at the diagnostic threshold: {_ratio_txt(d['fmr_diagnostic'])}.",
              f"- FNMR at the operating threshold: {_ratio_txt(d['fnmr'])}."]
    ind = b["race_1to1"].get("indian")
    if ind:
        lines += ["", f"**Indian group (deployment population):** only {ind['n_identities']} identities, "
                  f"{ind['n_genuine']} genuine and {ind['n_impostor']:,} impostor pairs in the test split. Zero errors here "
                  "means *not measurable*, not *no bias*: with this few pairs an FMR below "
                  f"{(ind.get('fmr_upper_if_zero') or 0):.1e} cannot be distinguished from zero, and the impostor tail "
                  f"(p99.9 {num(ind['impostor_p999'])}) rests on the top {max(1, int(ind['n_impostor'] * 0.001))} of "
                  f"{ind['n_impostor']:,} pairs. "
                  "An Indian-specific evaluation set is required before deployment."]
    lines += ["", img(b["plot"], "FMR / FNMR by group"), "",
              "Interpretation: with one global threshold, impostor pairs of two different Black or Asian people reach "
              "high similarity more often than pairs of two different white people, so those groups get more false "
              "alerts. Absolute FMR at the operating threshold is still small on LFW, but the ratio is what carries "
              "over to deployment. No mitigation was applied this week; this section only measures.", ""]
    return "\n".join(lines)


def sec_robustness(R):
    r = R["robustness"]
    if not r:
        return ""
    lines = ["## 7a. Robustness to image degradation", "", r["protocol"], "",
             f"Threshold {num(r['threshold'])}; {r['n_mated_probes']} mated + {r['n_nonmated_probes']} non-mated probes.", ""]
    for kind, rows in r["results"].items():
        lines += [f"**{kind}**", "", "| Level | FTE | FNIR [95% CI] | FPIR | Rank-1 (no threshold) |", "|---|---|---|---|---|"]
        for row in rows:
            lines.append(f"| {row['level']} | {pct(row['fte_rate'], 1)} | {pct(row['fnir'], 1)}{ci(row['fnir_ci'], 1)} | "
                         f"{pct(row['fpir'], 1)} | {pct(row['rank1_any_threshold'], 1)} |")
        lines += ["", img(f"eval/plots/robust_{kind}.png", f"robustness {kind}"), ""]
    m = r["minimum_usable_face_width"]
    lines.append(f"**Minimum usable face size: {m['width_px']} px** wide with the combined cctv degradation "
                 f"(clean downscaling alone stays usable down to {m.get('width_px_clean_downscale')} px). Rule: {m['rule']}; "
                 f"undegraded FNIR {pct(m['baseline_fnir'], 1)}. Faces smaller than this in a sighting photo should be "
                 "treated as unlikely to match even when the person is enrolled.")
    lines.append("")
    lines.append("Interpretation: on these clean, frontal LFW probes the model tolerates JPEG compression, brightness "
                 "changes and mild blur with no measurable loss at the operating threshold. The failures come from "
                 "very low resolution combined with noise and compression, strong blur (sigma 5 on a ~100 px face), "
                 "and covering the lower half of the face (masks), where detection itself starts to fail. Only "
                 f"{r['n_mated_probes']} mated probes were used, so a 1% FNIR means 2 misses; small differences "
                 "between levels are noise.")
    lines.append("")
    return "\n".join(lines)


def sec_quality(R):
    q = R["quality"]
    if not q:
        return ""
    rec = q["recommended_DET_SCORE_MIN"]
    lines = ["## 7b. Quality gate (DET_SCORE_MIN)", "", q["protocol"], "",
             f"Mean FNIR over the error-versus-reject curve (0-50% rejected): det_score {num(q['erc_mean_fnir']['det_score'], 4)}, "
             f"combined score {num(q['erc_mean_fnir']['combined'], 4)} -> combined is "
             f"{'better' if q['combined_beats_det_score'] else 'not better'} than det_score alone.", "",
             img(q["plot"], "Error versus reject"), "",
             "| Cut-off | Rejected (all) | Rejected (undegraded) | FNIR (rejects = misses) | FPIR |", "|---|---|---|---|---|"]
    for s in q["det_score_sweep"]:
        lines.append(f"| {s['cutoff']:.2f} | {pct(s['rejected_fraction'], 1)} | {pct(s['rejected_fraction_undegraded'], 1)} | "
                     f"{pct(s['fnir_overall'], 1)} | {pct(s['fpir_overall'], 2)} |")
    lines += ["", f"**Recommended DET_SCORE_MIN = {rec['value']}** ({rec['rule']}).", "", q.get("note", ""), ""]
    return "\n".join(lines)


def sec_detection(R):
    d = R["detection_widerface"]
    if not d:
        return ""
    lines = ["## 7c. Crowd face detection (WIDER FACE val)", "", d["protocol"], "",
             "| det_size | AP Easy | AP Medium | AP Hard | Recall Hard @ score 0.5 | s/image (mean, p95) |", "|---|---|---|---|---|---|"]
    for size, r in d["results"].items():
        lines.append(f"| {size} | {pct(r['ap']['easy'], 1)} | {pct(r['ap']['medium'], 1)} | {pct(r['ap']['hard'], 1)} | "
                     f"{pct(r['recall_at_det_thresh_0.5']['hard'], 1)} | {r['sec_per_image_mean']:.2f}, {r['sec_per_image_p95']:.2f} |")
    lines += ["", f"CPU: {d['cpu']}. Best Hard-subset AP: det_size {d.get('best_hard_det_size')}. "
              f"**Recommended det_size: {d['recommended_det_size']}**: {d.get('recommendation_reason', '')}.", ""]
    dc = R.get("detsize_check")
    if dc:
        lines += [f"det_size check on {dc['n_images']} LFW images: failure to enrol {pct(dc['fte_640'], 1)} at 640 vs "
                  f"{pct(dc['fte_' + str(dc['det_size'])], 1)} at {dc['det_size']}. Large faces scaled past the detector's "
                  "range are missed. Recommendation for later: run detection at two scales (640 and 1024) and merge, "
                  "so crowd photos get the Hard-subset gain without losing close-ups.", ""]
    return "\n".join(lines)


def sec_latency(R):
    lat = R["latency"]
    if not lat:
        return ""
    lines = ["## 7d. Latency (CPU)", "", lat["protocol"], "",
             f"CPU: {lat['cpu']}, {lat['logical_cpus']} logical CPUs, onnxruntime {lat['meta']['onnxruntime_version']} "
             f"({lat['onnxruntime_threads']}).", "",
             "| Faces in photo | Detected | process_frame p50 / p95 | detection p50 | embedding p50 | other models p50 | search p50 |",
             "|---|---|---|---|---|---|---|"]
    for n, r in lat["process_frame"].items():
        lines.append(f"| {n} | {r['faces_detected']} | {r['process_frame']['p50_ms']:.0f} / {r['process_frame']['p95_ms']:.0f} ms | "
                     f"{r['detection']['p50_ms']:.0f} ms | {r['embedding']['p50_ms']:.0f} ms | {r['other_models']['p50_ms']:.0f} ms | "
                     f"{r['search']['p50_ms']:.1f} ms |")
    by = lat.get("process_frame_by_det_size", {})
    if len(by) > 1:
        lines += ["", "process_frame p50 / p95 by detector input size:", "",
                  "| Faces | " + " | ".join(f"det_size {k}" for k in by) + " |", "|---|" + "---|" * len(by)]
        for n in next(iter(by.values())):
            lines.append(f"| {n} | " + " | ".join(f"{v[n]['process_frame']['p50_ms']:.0f} / {v[n]['process_frame']['p95_ms']:.0f} ms "
                                                 f"({v[n]['faces_detected']} found)" for v in by.values()) + " |")
    lines += ["", "Some LFW tiles contain bystanders, so more faces are detected than tiles. About half of the per-face "
              "time goes to the landmark and gender/age models that the app loads but does not use for matching; loading "
              "only detection + recognition (as the evaluation does) would roughly halve the latency of crowd photos.", ""]
    lines += ["", "| FAISS gallery | search p50 | p95 |", "|---|---|---|"]
    for n, r in lat["faiss_search"].items():
        lines.append(f"| {int(n):,} | {r['p50_ms']:.3f} ms | {r['p95_ms']:.3f} ms |")
    lines.append("")
    return "\n".join(lines)


def sec_label_noise(R):
    ln = R.get("label_noise")
    if not ln:
        return ""
    lines = ["## Label noise and look-alikes", "", ln["protocol"], "",
             "Highest non-mated open-set searches (test split, gallery 1000):", "",
             "| Probe identity | Rank-1 identity | Score |", "|---|---|---|"]
    lines += [f"| {r['probe_identity']} | {r['matched_identity']} | {r['score']:.3f} |" for r in ln["openset_top_nonmated_test"][:5]]
    lines += ["", f"Highest impostor pairs among {ln['n_impostor_pairs_test']:,} test-split pairs (first image per identity):", "",
              "| Identity A | Identity B | Score |", "|---|---|---|"]
    lines += [f"| {r['identity_a']} | {r['identity_b']} | {r['score']:.3f} |" for r in ln["impostor_top_pairs_test"][:5]]
    lines += ["", "Reading the names in this run: Talisa Soto / Talisa Bratt, Doc Rivers / Glenn Rivers and Odai / Uday "
              "Hussein are the same people under two names; scores of 1.000 and 0.958 mean the same (or a nearly "
              "identical) photo is filed under two people; Ashley and Mary-Kate Olsen are identical twins, a true "
              "look-alike. Six duplicate names with matching tokens or obvious spelling variants were merged before any "
              "experiment (see `eval/config/openset.yaml`).", ""]
    return "\n".join(lines)


THREATS = """## Threats to validity

- **LFW label noise.** Some people appear under two names and some photos are filed under the wrong
  person (see "Label noise and look-alikes"). They inflate non-mated scores, mostly at FPIR 0.1%.

- **LFW is optimistic.** It is celebrity photos: mostly frontal, well lit, high resolution and taken by
  professionals. Sighting photos from phones and CCTV will produce lower genuine scores, so real FNIR
  will be higher than the LFW numbers at the same threshold.
- **Synthetic degradations are not real CCTV.** Downscaling, blur, JPEG, gamma and a grey rectangle
  approximate some effects of real cameras, but not motion blur, rolling shutter, extreme pose, night
  IR, or compression artefacts of real video pipelines.
- **Demographic labels are noisy** (classifier scores, not self-reported), the groups are unbalanced,
  and the Indian group, which is the deployment population, has very little data.
- **No children and no large age gaps.** Missing-person cases often involve children and photos taken
  years apart. Neither is covered yet (Week 3).
- **One reference photo per case** is assumed. Several reference photos per case would change the
  numbers (usually for the better).
- **Small non-mated sets.** FPIR = 0.1% rests on a handful of errors; the bootstrap CIs show this.
"""


def build_report():
    names = ["embed_lfw", "verification_lfw", "openset_lfw", "operating_point", "calibration", "bias",
             "robustness", "quality", "detection_widerface", "latency", "detsize_check", "label_noise"]
    R = {n: _load(n) for n in names}
    meta = next((r["meta"] for r in R.values() if r), {})
    parts = [
        "# LocateNet evaluation report", "",
        "Generated by `python -m eval report` from `eval/results/*.json`. Do not edit by hand; re-run the experiments "
        "(`make eval-all`) instead.", "",
        f"Model: InsightFace `{meta.get('model', 'buffalo_l')}` (SCRFD-10GF detector + ArcFace ResNet-50 trained on "
        f"WebFace600K), insightface {meta.get('insightface_version')}, onnxruntime {meta.get('onnxruntime_version')}. "
        "Each result file records its git commit, dataset version and timestamp.", "",
        "## Headline", "", headline(R), "",
        "CIs are 95% bootstrap intervals that resample identities. Scores are cosine similarities of L2-normalised "
        "embeddings; FNIR/FPIR follow the NIST FRVT 1:N definitions at rank 1.", "",
        sec_embed(R), sec_verification(R), sec_openset(R), sec_operating(R), sec_calibration(R), sec_bias(R),
        sec_label_noise(R), sec_robustness(R), sec_quality(R), sec_detection(R), sec_latency(R), THREATS,
        "## Reproduce", "", "```bash",
        "pip install -r requirements.txt -r requirements-eval.txt",
        "python scripts/get_eval_data.py",
        "make eval-all        # or: python -m eval run eval/config/<experiment>.yaml",
        "python -m eval report", "```", "",
    ]
    path = os.path.join(ROOT, "docs", "EVALUATION.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(p for p in parts if p is not None))
    print(f"Wrote {os.path.relpath(path, ROOT)}")
    build_model_card(R, meta)
    update_readme(R)


README_START, README_END = "<!-- results:start -->", "<!-- results:end -->"


def update_readme(R):
    """Replace the generated block between the results markers in README.md."""
    path = os.path.join(ROOT, "README.md")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if README_START not in text or README_END not in text:
        print("README.md has no results markers; skipped")
        return
    block = "\n".join([
        README_START,
        "<!-- generated by `python -m eval report`; edit eval/report.py, not this block -->", "",
        headline(R), "",
        '<img src="eval/plots/openset_det.png" width="440"> <img src="eval/plots/bias_fmr_fnmr.png" width="560">', "",
        "Full protocols, CIs and plots: [docs/EVALUATION.md](docs/EVALUATION.md) · "
        "[docs/MODEL_CARD.md](docs/MODEL_CARD.md) · [docs/OPERATING_POINT.md](docs/OPERATING_POINT.md)",
        README_END,
    ])
    start, end = text.index(README_START), text.index(README_END) + len(README_END)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text[:start] + block + text[end:])
    print("Updated README.md results block")


def build_model_card(R, meta):
    op, b = R["operating_point"], R["bias"]
    rec = op["recommended"] if op else {}
    lines = [
        "# Model card: LocateNet face matching", "",
        "Format: Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019). Numbers are generated from "
        "`eval/results/*.json` by `python -m eval report`; see [EVALUATION.md](EVALUATION.md) for protocols and CIs.", "",
        "## Model details", "",
        f"- **Components:** InsightFace model pack `buffalo_l` (insightface {meta.get('insightface_version')}, onnxruntime "
        f"{meta.get('onnxruntime_version')}, CPU).",
        "  - Detector: SCRFD-10GF (`det_10g.onnx`), 5-point landmarks.",
        "  - Recognition: ArcFace loss, **ResNet-50** backbone trained on **WebFace600K** (`w600k_r50.onnx`), 512-d "
        "embedding. (Not R100.)",
        "  - The app also loads buffalo_l landmark and gender/age models; they are not used for matching.",
        "- **Matching:** cosine similarity of L2-normalised embeddings, exact search with FAISS `IndexFlatIP` + "
        "`IndexIDMap`, top-5, first active case at or above `MATCH_THRESHOLD` wins.",
        "- **Calibration:** rank-1 score mapped to an estimated P(correct match) by a monotone calibrator stored as "
        "JSON (`models/calibrator.json`), fitted at an explicit 10% mated prior.",
        "- **Developed by:** LocateNet project (faizdevx); model weights by the InsightFace project (non-commercial "
        "research licence of the buffalo_l pack applies).", "",
        "## Intended use", "",
        "- **Primary use:** generate *leads* for missing-person cases. A sighting photo is compared with the reference "
        "photos of open cases; possible matches are shown to the officer who owns the case, who must confirm or reject "
        "every one.",
        "- **Users:** trained police / NGO case officers see and review matches. The public uploads photos; after "
        "an upload the app tells them how many matches were found, and a hotspot message names the matched case "
        "(a privacy issue noted for review).",
        "- **Human in the loop is mandatory.** No action against a person may be taken on a match alone.", "",
        "## Out-of-scope uses", "",
        "- Surveillance, tracking or profiling of people who are not missing.",
        "- Automated decisions of any kind (arrest, detention, denial of services) without human review.",
        "- Identifying protesters, journalists, or members of any group; scanning crowds for watch lists.",
        "- Any use on children or across large age gaps until those are evaluated (Week 3).",
        "- Commercial use (the buffalo_l weights are licensed for non-commercial research).", "",
        "## Factors", "",
        "- **Relevant factors:** face size in pixels, blur, compression, lighting, occlusion (masks), pose, age, and "
        "demographic group. Robustness and bias results are in EVALUATION.md sections 6 and 7.",
        "- **Evaluation factors reported:** race group (4 groups) and gender (LFW-attribute labels), image degradations, "
        "gallery size.", "",
        "## Metrics", "",
        "- 1:1 verification accuracy and TAR@FAR on LFW (sanity check).",
        "- 1:N open-set FNIR at FPIR = 0.1% / 1% / 5% (NIST FRVT definitions), rank-1 / rank-5 rates.",
        "- Calibration: ECE and Brier score at a stated prior.",
        "- Per-group FMR / FNMR at the single global threshold, and max/min disparity ratios.",
        "- All with 95% bootstrap CIs over identities.", "",
        "## Operating point", "",
        f"- `MATCH_THRESHOLD = {rec.get('MATCH_THRESHOLD')}` (FPIR {pct(rec.get('match_fpir'), 1)} on the tune split), "
        f"`HIGH_CONF_THRESHOLD = {rec.get('HIGH_CONF_THRESHOLD')}` (FPIR {pct(rec.get('high_fpir'), 1)}). "
        "Reasoning: [OPERATING_POINT.md](OPERATING_POINT.md).", "",
        "## Evaluation data", "",
        "- **LFW** (13,233 images, 5,749 people): verification, open-set identification, calibration, robustness, "
        "bias fallback. Split by identity into tune / test halves.",
        "- **LFW attributes** (Kumar et al.) for demographic groups (fallback for RFW).",
        "- **WIDER FACE val** (3,226 images, 39,708 faces) for crowd detection.",
        "- **Not yet:** RFW (licence pending), CCTV footage, Indian-population data, children, age-gap data.", "",
        "## Training data", "",
        "Not trained or fine-tuned by this project. The recognition model was trained by InsightFace on WebFace600K; "
        "its demographic composition is not documented in detail by the authors.", "",
        "## Quantitative analyses", "",
        headline(R), "",
        "## Ethical considerations", "",
    ]
    if b:
        d = b["disparity"]
        src = "LFW-attribute fallback, weak evidence" if "FALLBACK" in b.get("data", "") else "RFW"
        lines += [f"- Bias audit ({src}), one global threshold. Asian and Black identities get more false matches "
                  "than white identities; no mitigation has been applied. Max/min ratios:",
                  f"  - FMR at the diagnostic threshold: {_ratio_txt(d['fmr_diagnostic'])}",
                  f"  - FMR at the operating threshold: {_ratio_txt(d['fmr'])}",
                  f"  - FNMR at the operating threshold: {_ratio_txt(d['fnmr'])}"]
        ind = b["race_1to1"].get("indian")
        if ind:
            lines.append(f"- The deployment population (India) is represented by only {ind['n_identities']} identities in "
                         "the test data. The system is **not validated** for Indian faces.")
    lines += [
        "- A false match can direct police attention at an innocent person; every alert says it needs human "
        "confirmation and shows an *estimated* likelihood, not a certainty.",
        "- Sighting photos contain bystanders. Their faces are embedded and stored as crops; retention rules apply "
        "(see README, DPDP Act 2023).",
        "- Evaluation images are of public figures (LFW); no evaluation images or embeddings are committed.", "",
        "## Caveats and recommendations", "",
        "- LFW is easier than real sightings; expect higher miss rates in deployment.",
        "- Re-run the operating-point experiment when the number of open cases changes by 10x.",
        "- Obtain RFW and an Indian evaluation set before any field use; add child and age-gap evaluation (Week 3).",
        "- Prefer several reference photos per case when available.", "",
    ]
    path = os.path.join(ROOT, "docs", "MODEL_CARD.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"Wrote {os.path.relpath(path, ROOT)}")
