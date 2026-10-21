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
