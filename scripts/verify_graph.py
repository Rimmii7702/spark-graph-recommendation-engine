"""Independently verify graph results using only the Python standard library.

This recomputes degrees, weak components, undirected triangles and fixed-iteration
PageRank from the original Wikipedia vote network. Full-data verification may
take several minutes on a laptop.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def read_edges(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    edges = set()
    with opener(path, "rt", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if len(tokens) < 2:
                continue
            try:
                edges.add((int(tokens[0]), int(tokens[1])))
            except ValueError:
                continue
    return edges


def degree_top(edges: set[tuple[int, int]]):
    out = Counter(src for src, _ in edges)
    ins = Counter(dst for _, dst in edges)
    return ([{"id": node, "outDegree": n} for node, n in
             sorted(out.items(), key=lambda z: (-z[1], z[0]))[:5]],
            [{"id": node, "inDegree": n} for node, n in
             sorted(ins.items(), key=lambda z: (-z[1], z[0]))[:5]])


def undirected_neighbors(edges):
    neighbors = {n: set() for edge in edges for n in edge}
    for src, dst in edges:
        if src != dst:
            neighbors[src].add(dst)
            neighbors[dst].add(src)
    return neighbors


def component_sizes(neighbors):
    unseen = set(neighbors)
    components = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        pending = [start]
        size = 0
        while pending:
            node = pending.pop()
            size += 1
            found = neighbors[node] & unseen
            unseen.difference_update(found)
            pending.extend(found)
        components.append({"component": start, "count": size})
    return sorted(components, key=lambda row: (-row["count"], row["component"]))


def triangle_counts(neighbors):
    # Orient each undirected edge by vertex ID. Each a < b < c triangle
    # appears exactly once, including graphs with reciprocal directed edges.
    forward = {n: {v for v in adjacent if v > n} for n, adjacent in neighbors.items()}
    counts = {n: 0 for n in neighbors}
    for a, larger in forward.items():
        for b in larger:
            for c in larger & forward[b]:
                counts[a] += 1
                counts[b] += 1
                counts[c] += 1
    return counts


def fixed_pagerank(edges, iterations, reset_probability=0.15):
    """GraphX fixed-iteration semantics, including final sum-to-N scaling.

    Start at 1; update alpha + (1-alpha)*incoming weighted ranks each round.
    Do not redistribute dangling mass every round (a different PageRank variant).
    """
    if iterations < 1:
        raise ValueError("PageRank iterations must be positive")
    ordered = sorted(edges)
    nodes = sorted({n for edge in edges for n in edge})
    degrees = Counter(src for src, _ in ordered)
    ranks = dict.fromkeys(nodes, 1.0)
    for _ in range(iterations):
        incoming = dict.fromkeys(nodes, 0.0)
        for src, dst in ordered:
            incoming[dst] += ranks[src] / degrees[src]
        ranks = {n: reset_probability + (1 - reset_probability) * incoming[n] for n in nodes}
    scale = len(nodes) / sum(ranks.values())
    return {n: rank * scale for n, rank in ranks.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--verification-output", type=Path,
                   help="Defaults to verification.json beside the actual results")
    p.add_argument("--expect-wiki-vote", action="store_true",
                   help="Also compare with SNAP's published full-network statistics")
    args = p.parse_args()
    edges = read_edges(args.input)
    nodes = {n for src, dst in edges for n in (src, dst)}
    results = json.loads(args.results.read_text(encoding="utf-8"))
    checks = {
        "number of vertices": len(nodes) == results["vertex_count"],
        "number of directed edges": len(edges) == results["edge_count"],
    }
    out, ins = degree_top(edges)
    checks["top outdegree values"] = out == results["a_outdegree"]
    checks["top indegree values"] = ins == results["b_indegree"]
    neighbors = undirected_neighbors(edges)
    components = component_sizes(neighbors)
    checks["top weakly connected components"] = components[:5] == results["d_components"]
    triangles = triangle_counts(neighbors)
    triangle_top = [{"id": n, "count": count} for n, count in
                    sorted(triangles.items(), key=lambda pair: (-pair[1], pair[0]))[:5]]
    checks["top triangle counts"] = triangle_top == results["e_triangles"]
    ranks = fixed_pagerank(edges, results["pagerank_iterations"])
    rank_top = sorted(ranks, key=lambda n: (-ranks[n], n))[:5]
    actual_ranks = results["c_pagerank"]
    checks["top PageRank IDs"] = rank_top == [row["id"] for row in actual_ranks]
    checks["PageRank numeric values"] = len(actual_ranks) == len(rank_top) and all(
        row["id"] in ranks and math.isclose(row["pagerank"], ranks[row["id"]],
                                           rel_tol=1e-8, abs_tol=1e-10)
        for row in actual_ranks)
    total_triangles = sum(triangles.values()) // 3
    if args.expect_wiki_vote:
        checks["SNAP published vertex/edge counts"] = (len(nodes), len(edges)) == (7115, 103689)
        checks["SNAP published largest WCC"] = components[0]["count"] == 7066
        checks["SNAP published total triangles"] = total_triangles == 608389
    report = {"verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "vertex_count": len(nodes), "edge_count": len(edges),
              "component_count": len(components), "total_undirected_triangles": total_triangles,
              "checks": checks}
    output = args.verification_output or args.results.with_name("verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for check, ok in checks.items():
        print(f'{"PASS" if ok else "FAIL"}: {check}')
    if not all(checks.values()):
        raise SystemExit("Cross-validation failure: inspect the graph analytics outputs.")
    print(f"All five queries independently verified; {len(components)} components, "
          f"{total_triangles:,} undirected triangles. Saved {output}")


if __name__ == "__main__":
    main()
