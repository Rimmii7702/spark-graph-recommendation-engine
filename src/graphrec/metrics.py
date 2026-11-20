"""Pure ranking metrics and recommendation comparison helpers."""
from __future__ import annotations


def evaluate_ranking(
    recommendations: dict[int, list[int]],
    held_out_friends: dict[int, set[int]],
    k: int = 10,
) -> dict:
    """Macro ranking metrics over users with >=1 held-out positive edge.

    Precision@k = hits / k, even for shorter output lists.
    Recall@k = hits / number of withheld friends.
    HitRate@k = fraction of evaluated users with >=1 hit.
    These are evaluations against withheld graph links, not online CTR.
    """
    if k < 1:
        raise ValueError("k must be positive")
    users = sorted(user for user, friends in held_out_friends.items() if friends)
    if not users:
        return {"evaluated_users": 0, "precision_at_k": None, "recall_at_k": None,
                "hit_rate_at_k": None, "coverage": None, "hits": 0,
                "held_out_links_for_evaluated_users": 0, "k": k}
    hits_total = 0
    precision_total = 0.0
    recall_total = 0.0
    hit_users = 0
    covered_users = 0
    total_truth = 0
    for user in users:
        truth = held_out_friends[user]
        total_truth += len(truth)
        # De-duplicate candidate IDs without changing their original order.
        ranked = list(dict.fromkeys(recommendations.get(user, [])))[:k]
        hits = len(set(ranked) & truth)
        hits_total += hits
        precision_total += hits / k
        recall_total += hits / len(truth)
        hit_users += int(hits > 0)
        covered_users += int(bool(ranked))
    n = len(users)
    return {
        "evaluated_users": n,
        "precision_at_k": round(precision_total / n, 6),
        "recall_at_k": round(recall_total / n, 6),
        "hit_rate_at_k": round(hit_users / n, 6),
        "coverage": round(covered_users / n, 6),
        "hits": hits_total,
        "held_out_links_for_evaluated_users": total_truth,
        "k": k,
    }


def compare_recommendations(a: dict[int, list[int]], b: dict[int, list[int]], k: int = 10) -> dict:
    users = sorted(set(a) & set(b))
    common = [len(set(a[user][:k]) & set(b[user][:k])) for user in users]
    return {
        "compared_users": len(users),
        "mean_overlap_at_k": round(sum(common) / len(common), 4) if common else None,
        "total_overlap_at_k": sum(common),
    }
