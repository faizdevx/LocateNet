# Model card: LocateNet face matching

Format: Mitchell et al., *Model Cards for Model Reporting* (FAT* 2019). Numbers are generated from `eval/results/*.json` by `python -m eval report`; see [EVALUATION.md](EVALUATION.md) for protocols and CIs.

## Model details

- **Components:** InsightFace model pack `buffalo_l` (insightface 2.1, onnxruntime 1.31.0, CPU).
  - Detector: SCRFD-10GF (`det_10g.onnx`), 5-point landmarks.
  - Recognition: ArcFace loss, **ResNet-50** backbone trained on **WebFace600K** (`w600k_r50.onnx`), 512-d embedding. (Not R100.)
  - The app also loads buffalo_l landmark and gender/age models; they are not used for matching.
- **Matching:** cosine similarity of L2-normalised embeddings, exact search with FAISS `IndexFlatIP` + `IndexIDMap`, top-5, first active case at or above `MATCH_THRESHOLD` wins.
- **Calibration:** rank-1 score mapped to an estimated P(correct match) by a monotone calibrator stored as JSON (`models/calibrator.json`), fitted at an explicit 10% mated prior.
- **Developed by:** LocateNet project (faizdevx); model weights by the InsightFace project (non-commercial research licence of the buffalo_l pack applies).

## Intended use

- **Primary use:** generate *leads* for missing-person cases. A sighting photo is compared with the reference photos of open cases; possible matches are shown to the officer who owns the case, who must confirm or reject every one.
- **Users:** trained police / NGO case officers see and review matches. The public uploads photos; after an upload the app tells them how many matches were found, and a hotspot message names the matched case (a privacy issue noted for review).
- **Human in the loop is mandatory.** No action against a person may be taken on a match alone.

## Out-of-scope uses

- Surveillance, tracking or profiling of people who are not missing.
- Automated decisions of any kind (arrest, detention, denial of services) without human review.
- Identifying protesters, journalists, or members of any group; scanning crowds for watch lists.
- Any use on children or across large age gaps until those are evaluated (Week 3).
- Commercial use (the buffalo_l weights are licensed for non-commercial research).

## Factors

- **Relevant factors:** face size in pixels, blur, compression, lighting, occlusion (masks), pose, age, and demographic group. Robustness and bias results are in EVALUATION.md sections 6 and 7.
- **Evaluation factors reported:** race group (4 groups) and gender (LFW-attribute labels), image degradations, gallery size.

## Metrics

- 1:1 verification accuracy and TAR@FAR on LFW (sanity check).
- 1:N open-set FNIR at FPIR = 0.1% / 1% / 5% (NIST FRVT definitions), rank-1 / rank-5 rates.
- Calibration: ECE and Brier score at a stated prior.
- Per-group FMR / FNMR at the single global threshold, and max/min disparity ratios.
- All with 95% bootstrap CIs over identities.

## Operating point

- `MATCH_THRESHOLD = 0.312` (FPIR 1.0% on the tune split), `HIGH_CONF_THRESHOLD = 0.634` (FPIR 0.1%). Reasoning: [OPERATING_POINT.md](OPERATING_POINT.md).

## Evaluation data

- **LFW** (13,233 images, 5,749 people): verification, open-set identification, calibration, robustness, bias fallback. Split by identity into tune / test halves.
- **LFW attributes** (Kumar et al.) for demographic groups (fallback for RFW).
- **WIDER FACE val** (3,226 images, 39,708 faces) for crowd detection.
- **Not yet:** RFW (licence pending), CCTV footage, Indian-population data, children, age-gap data.

## Training data

Not trained or fine-tuned by this project. The recognition model was trained by InsightFace on WebFace600K; its demographic composition is not documented in detail by the authors.

## Quantitative analyses

| Metric | Value [95% CI] |
|---|---|
| LFW 1:1 accuracy (10-fold), enrolled pairs / end to end | 99.83% ± 0.20% / 99.82% ± 0.19% |
| FNIR @ FPIR=1%, gallery 1000 (test) | 0.44% [0.07%, 0.87%] |
| Rank-1 identification, gallery 1000 (test) | 99.68% [99.29%, 99.94%] |
| ECE raw (min-max) -> calibrated (platt) | 0.125 -> 0.006 |
| Worst/best group FMR ratio at pooled FMR 1e-4 (LFW fallback; Indian group too small to measure) | 9.04x (asian vs white) [5.94, 15.67] |
| FNMR, child enrolment vs 0-2 y / 21+ y age gap (FG-NET) | 2.8% / 59.2%  [52.0%, 66.7%] |
| Minimum usable face width | 32 px |
| p95 latency, 10-face photo (CPU, det_size 640) | 2588 ms |

## Ethical considerations

- Bias audit (LFW-attribute fallback, weak evidence), one global threshold. Asian and Black identities get more false matches than white identities; no mitigation has been applied. Max/min ratios:
  - FMR at the diagnostic threshold: unbounded (indian has zero errors); among groups with errors: **9.04x** (asian vs white), 95% CI 5.94-15.67
  - FMR at the operating threshold: unbounded (indian has zero errors); among groups with errors: **14.00x** (black vs white), 95% CI 6.10-40.88
  - FNMR at the operating threshold: unbounded (black has zero errors)
- The deployment population (India) is represented by only 58 identities in the test data. The system is **not validated** for Indian faces.
- A false match can direct police attention at an innocent person; every alert says it needs human confirmation and shows an *estimated* likelihood, not a certainty.
- Sighting photos contain bystanders. Their faces are embedded and stored as crops; retention rules apply (see README, DPDP Act 2023).
- Evaluation images are of public figures (LFW); no evaluation images or embeddings are committed.

## Caveats and recommendations

- LFW is easier than real sightings; expect higher miss rates in deployment.
- Re-run the operating-point experiment when the number of open cases changes by 10x.
- Obtain RFW and an Indian evaluation set before any field use; add child and age-gap evaluation (Week 3).
- Prefer several reference photos per case when available.
