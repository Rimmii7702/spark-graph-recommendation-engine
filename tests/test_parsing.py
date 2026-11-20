import pytest
from graphrec.data import parse_adjacency, parse_edge_line, to_canonical_pairs


def test_adjacency_parse_and_self_links():
    assert parse_adjacency('5\t3,7,5,3') == [(5, frozenset({3, 7}))]
    assert parse_adjacency('5\t') == [(5, frozenset())]
    assert parse_adjacency('# description') == []
    assert parse_adjacency('  ') == []


def test_canonical_pairs():
    assert set(to_canonical_pairs((5, frozenset({2, 8})))) == {(2, 5), (5, 8)}


def test_reject_invalid_ids():
    with pytest.raises(ValueError):
        parse_adjacency('foo\t10')
    with pytest.raises(ValueError):
        parse_adjacency('3\t-1')
    with pytest.raises(ValueError):
        parse_adjacency('2147483648\t1')


def test_directed_edges():
    assert parse_edge_line('# comments') == []
    assert parse_edge_line('12 33') == [(12, 33)]
    with pytest.raises(ValueError):
        parse_edge_line('12')
