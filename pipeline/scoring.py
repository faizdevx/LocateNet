import config

CATEGORY_HIGH = "High Similarity"
CATEGORY_POSSIBLE = "Possible Match – Needs Review"
CATEGORY_NONE = "No Match"


def categorize(similarity):
    """Map a raw cosine similarity (0-1) to a review category."""
    if similarity >= config.HIGH_CONF_THRESHOLD:
        return CATEGORY_HIGH
    if similarity >= config.MATCH_THRESHOLD:
        return CATEGORY_POSSIBLE
    return CATEGORY_NONE


def select_match(scores, ids, is_active=None):
    """
    Pick the first top-k neighbour that is above the match threshold and whose case is active.
    FAISS always returns a nearest neighbour, so the threshold is what decides a match.
    Returns (case_id, similarity); case_id is -1 when nothing qualifies, and similarity is then
    the best score among active candidates (0.0 if there are none).
    """
    best_active = 0.0
    for score, case_id in zip(scores, ids):
        case_id = int(case_id)
        if case_id == -1:
            continue
        if is_active is not None and not is_active(case_id):
            continue
        score = float(score)
        if score >= config.MATCH_THRESHOLD:
            return case_id, score
        best_active = max(best_active, score)
        break
    return -1, best_active


def match_fields(scores, ids, is_active=None):
    """Shared match output used by both the image and the video pipeline."""
    case_id, similarity = select_match(scores, ids, is_active)
    return {
        "id": case_id,
        "label": f"Person_{case_id}" if case_id != -1 else "Unknown",
        "similarity": similarity,
        "percentage": round(similarity * 100, 2),
        "category": categorize(similarity) if case_id != -1 else CATEGORY_NONE,
    }


def rank_cases(embedding, search_svc, matcher=None, top_k=None):
    """
    Case-level (scores, case_ids). FAISS stores one vector per reference photo; with a
    TemplateMatcher the per-photo hits (top_k * MAX_PHOTOS_PER_CASE of them) are grouped by case and
    re-scored with the configured strategy. Without a matcher the FAISS ids are used as they are
    (evaluation protocols that enrol one vector per identity).
    """
    top_k = int(top_k or config.TOP_K)
    if matcher is None:
        return search_svc.search_face(embedding, top_k=top_k)
    hits = search_svc.search_face(embedding, top_k=matcher.search_k(top_k))
    return matcher.rank(embedding, hits, top_k=top_k)
