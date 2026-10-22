"""Experiment `age_mitigation`: does up-weighting recent reference photos help aged probes?

Setup (FG-NET, same subject split as multi_template): every gallery subject enrols all of its
earlier photos (up to 5, age-diverse when it has more); probes are its latest third of photos.
Non-mated probes: held-out FG-NET subjects + LFW non-mated test probes; LFW single-image
distractors pad the gallery. A photo's age at search time is probe_age - photo_age, so the
weights are computed per probe (in deployment: years since the photo date).
Variants: strategy in {max, quality_weighted_mean} x tau in {none, 2, 5, 10} years
(services/template_matcher: weights exp(-years/tau); for max a penalty of
recency_penalty * (1 - weight)). Threshold per variant at FPIR = 1% on tune, FNIR on test,
bootstrap CI over subjects. A variant is shipped only if its test FNIR CI lies entirely below
the no-recency baseline of the same strategy.
"""
import numpy as np
import pandas as pd

from eval import metrics as M
from eval.common import DATA_DIR, metadata, save_result
from eval.datasets import fgnet
from eval.embed import STATUS_OK, cache_name, embed_paths
from eval.protocols import openset
from eval.protocols.multi_template import age_diverse
from services.template_matcher import CaseTemplate, case_score

TAUS = [None, 2, 5, 10]
STRATS = ["max", "quality_weighted_mean"]


def build(cfg, seed):
    man = fgnet.load_manifest(DATA_DIR)
    out = embed_paths(list(man["image_path"]), cache_name("fgnet"))
    emb_of = {p: e for p, e, st in zip(man["image_path"], out["emb"], out["status"]) if st == STATUS_OK}
    q = {p: float(d) for p, d in zip(man["image_path"], out["det_score"])}
    age = dict(zip(man["image_path"], fgnet.ages(man)))
    _, _, lfw_emb, proto = openset.load_lfw_protocol({"seed": 0, "nonmated_frac": 0.3, "max_probes_per_identity": 10,
                                                       "identity_aliases": cfg.get("identity_aliases")})
    rng = np.random.default_rng(seed + 53)  # same subject permutation as multi_template
    subjects = list(rng.permutation(sorted(man["identity_id"].unique())))
    splits = {"tune": subjects[: len(subjects) // 2], "test": subjects[len(subjects) // 2:]}
    data = {}
    for split, subs in splits.items():
        subs = list(rng.permutation(subs))
        n_non = int(round(0.3 * len(subs)))
        non, gal = subs[:n_non], subs[n_non:]
        templates, probes = {}, []
        for s in gal:
            photos = sorted((p for p in man.loc[man["identity_id"] == s, "image_path"] if p in emb_of), key=lambda p: (age[p], p))
            n_probe = max(1, len(photos) // 3)
            pool = age_diverse(photos[:-n_probe], int(cfg.get("max_photos", 5)))
            if pool:
                templates[s] = pool
                probes += [(p, s, True, age[p]) for p in photos[-n_probe:]]
        for s in non:
            probes += [(p, s, False, age[p]) for p in man.loc[man["identity_id"] == s, "image_path"] if p in emb_of]
        lfw_non = proto[split]["probes"]
        lfw_non = lfw_non[(~lfw_non["mated"]) & lfw_non["image_path"].isin(lfw_emb)]
        probes += [(p, i, False, np.nan) for p, i in zip(lfw_non["image_path"], lfw_non["identity_id"])]
        distractors = {i: p for i, p in proto[split]["distractors"].items() if p in lfw_emb}
        data[split] = {"templates": templates, "probes": probes, "distractors": distractors}
    return data, {**lfw_emb, **emb_of}, q, age


def score_split(d, emb, q, age, strategy, tau):
    names = list(d["templates"]) + list(d["distractors"])
    dist_mat = np.stack([emb[p] for p in d["distractors"].values()]) if d["distractors"] else np.zeros((0, 512))
    recs = []
    for p, ident, mated, probe_age in d["probes"]:
        e = emb[p]
        scores = []
        for s, photos in d["templates"].items():
            years = np.array([max(0.0, probe_age - age[x]) if np.isfinite(probe_age) else np.nan for x in photos])
            t = CaseTemplate(0, np.stack([emb[x] for x in photos]), np.array([q[x] for x in photos]), years)
            scores.append(case_score(t, e, strategy, tau=tau))
        scores += list(dist_mat @ e)  # single-photo distractors: every strategy reduces to the cosine
        scores = np.array(scores)
        best = int(np.argmax(scores))
        recs.append((ident, mated, float(scores[best]), 1 if names[best] == ident else 0))
    return pd.DataFrame(recs, columns=["identity_id", "mated", "top_score", "rank"])


def run(cfg):
    n_boot, seed = int(cfg.get("n_boot", 500)), int(cfg["seed"])
    data, emb, q, age = build(cfg, seed)
    res = {}
    for strategy in STRATS:
        res[strategy] = {}
        for tau in TAUS:
            tune = score_split(data["tune"], emb, q, age, strategy, tau)
            test = score_split(data["test"], emb, q, age, strategy, tau)
            t = M.threshold_at_rate(tune[~tune["mated"]]["top_score"].to_numpy(float), 0.01)
            units = openset.to_units(test)
            f = M.bootstrap_ci(lambda us: M.fnir_fpir_at_threshold(*openset.unit_arrays(us), t)[0], units, n=n_boot, seed=seed)
            fpir = M.fnir_fpir_at_threshold(*openset.unit_arrays(units), t)[1]
            res[strategy][str(tau)] = {"threshold_fpir1_tune": t, "test_fnir": f[0], "test_fnir_ci": [f[1], f[2]],
                                       "test_fpir": fpir, "test_rank1": float((test[test["mated"]]["rank"] == 1).mean())}
            print(f"{strategy:22s} tau={tau!s:4s} FNIR {f[0]:.3f} [{f[1]:.3f}, {f[2]:.3f}]  t={t:.3f}")
    decision = {}
    for strategy in STRATS:
        base = res[strategy]["None"]
        better = [tau for tau in TAUS[1:] if res[strategy][str(tau)]["test_fnir_ci"][1] < base["test_fnir"]]
        decision[strategy] = {"ship": bool(better), "taus_clearly_better": better,
                              "rule": "ship only if the variant's whole test FNIR CI is below the baseline FNIR"}
    sizes = {s: {"n_gallery_subjects": len(d["templates"]), "n_mated_probes": sum(1 for x in d["probes"] if x[2])}
             for s, d in data.items()}
    save_result("age_mitigation", {"meta": metadata(cfg, fgnet.DATASET_VERSION), "protocol": __doc__.strip().split("\n\n", 1)[1],
                                   "results": res, "decision": decision, "sizes": sizes,
                                   "adapter": "not attempted: needs CACD (~160k images, ~15 h of CPU embedding here) for "
                                              "identity-disjoint training; see docs/EVALUATION.md"})
    print("decision:", decision)
