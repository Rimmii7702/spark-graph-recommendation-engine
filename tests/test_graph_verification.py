"""Small graph checks for independent degree / triangle / components logic."""
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

spec = spec_from_file_location('graph_verifier', Path(__file__).resolve().parents[1] / 'scripts' / 'verify_graph.py')
v = module_from_spec(spec)
spec.loader.exec_module(v)


def test_independent_verifier():
    # Triangle 1-2-3, duplicated in reverse direction; plus isolated link 4-5.
    edges = {(1,2),(2,1),(2,3),(3,1),(4,5)}
    out, ins = v.degree_top(edges)
    assert out[0] == {'id':1,'outDegree':1} or out[0] == {'id':2,'outDegree':2}
    neighbors = v.undirected_neighbors(edges)
    assert v.component_sizes(neighbors)[0]['count'] == 3
    triangles = v.triangle_counts(neighbors)
    assert triangles[1] == 1 and triangles[2] == 1 and triangles[3] == 1
    assert triangles[4] == 0


def test_page_rank_normalization():
    ranks = v.fixed_pagerank({(1,2),(2,3),(3,1)}, 10)
    assert set(ranks) == {1,2,3}
    assert all(abs(score-1.0) < 1e-10 for score in ranks.values())
