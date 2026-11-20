from graphrec.metrics import evaluate_ranking, compare_recommendations


def test_metrics_all_hits():
    out = evaluate_ranking({1: [2, 3], 4: [1]}, {1: {2, 9}, 4: {1}}, k=2)
    assert out['evaluated_users'] == 2
    assert out['precision_at_k'] == 0.5
    assert out['recall_at_k'] == 0.75
    assert out['hit_rate_at_k'] == 1.0
    assert out['coverage'] == 1.0
    assert out['hits'] == 2


def test_zero_and_empty_truth():
    zero = evaluate_ranking({1: []}, {1: {2}}, k=10)
    assert zero['precision_at_k'] == 0.0
    assert zero['hit_rate_at_k'] == 0.0
    assert zero['coverage'] == 0.0
    empty = evaluate_ranking({1: [2]}, {1: set()}, k=10)
    assert empty['evaluated_users'] == 0
    assert empty['recall_at_k'] is None


def test_duplicate_and_overlap():
    out = evaluate_ranking({1: [2, 2, 4]}, {1: {2}}, k=3)
    assert out['hits'] == 1
    same = compare_recommendations({1: [2, 3], 5: []}, {1: [2, 4], 5: []})
    assert same['total_overlap_at_k'] == 1
    assert same['compared_users'] == 2
