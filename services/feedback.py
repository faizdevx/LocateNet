"""Safety rules for the human-feedback loop (pure functions, no database).

* Template growth: a face an officer confirmed may become an extra reference photo only if the
  officer owns the case (or is Admin), the case has fewer than MAX_SIGHTING_TEMPLATES such photos,
  the face passes the quality gate, and it looks like the person already enrolled (cosine >=
  MATCH_THRESHOLD to at least one enrolment photo). The last rule stops a wrongly confirmed
  stranger from being added to the case.
* False-positive suppression: faces marked false_positive become negative exemplars of that case.
  A later alert for the same case is DOWNGRADED (never dropped) when the probe is at least
  NEG_SUPPRESS_THRESHOLD similar to an exemplar and closer to it than to the case's best template.
"""
import numpy as np

DOWNGRADED_CATEGORY = "Needs Review – similar to a previously rejected face"


def _norm(v):
    v = np.asarray(v, dtype="float32")
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def template_growth_check(*, is_owner_or_admin, n_existing_sighting_templates, max_sighting_templates,
                          face_px, det_score, det_score_min, min_face_px, embedding, enrolment_embeddings,
                          match_threshold):
    """Return (allowed, reason). reason explains a refusal in words an officer can act on."""
    if not is_owner_or_admin:
        return False, "only the case's officer or an Admin can add reference photos"
    if n_existing_sighting_templates >= max_sighting_templates:
        return False, f"this case already has {max_sighting_templates} photos taken from sightings"
    if det_score is None or det_score < det_score_min:
        return False, "the face is too unclear to use as a reference photo"
    if face_px is None or face_px < min_face_px:
        return False, f"the face is too small ({0 if face_px is None else face_px:.0f} px, need {min_face_px:.0f} px)"
    if embedding is None or not len(enrolment_embeddings):
        return False, "no embedding available for this face"
    e = _norm(embedding)
    best = max(float(_norm(x) @ e) for x in enrolment_embeddings)
    if best < match_threshold:
        return False, (f"the face is not similar enough to the enrolled photos (best similarity {best:.2f} < "
                       f"{match_threshold:.2f}); it may be a different person")
    return True, None


def suppression_decision(probe_embedding, case_similarity, negative_embeddings, threshold):
    """(downgrade, best_negative_similarity). Downgrade only when the probe is closer to a rejected
    face of this case than to the case itself AND at least `threshold` similar to it."""
    if probe_embedding is None or not len(negative_embeddings):
        return False, None
    p = _norm(probe_embedding)
    neg = max(float(_norm(x) @ p) for x in negative_embeddings)
    return bool(neg >= threshold and neg > case_similarity), neg
