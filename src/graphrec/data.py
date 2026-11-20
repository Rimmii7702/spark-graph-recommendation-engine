"""Input normalization used by both Spark recommendation methods.

The LiveJournal-adjacency format is one row per user:
    user_id<TAB>friend_id,friend_id,...
An omitted/empty friend field is valid. Friendship is modeled as undirected;
original directed Wiki-Vote links are never passed through this parser.
"""
from __future__ import annotations


def parse_adjacency(line: str) -> list[tuple[int, frozenset[int]]]:
    line = line.strip("\r\n")
    if not line.strip() or line.lstrip().startswith("#"):
        return []
    parts = line.split("\t", 1)
    try:
        user = int(parts[0].strip())
        friends = frozenset(
            int(item.strip()) for item in (parts[1] if len(parts) > 1 else "").split(",")
            if item.strip()
        )
    except ValueError as exc:
        raise ValueError(f"Invalid adjacency-list row: {line[:120]!r}") from exc
    if user < 0 or any(friend < 0 for friend in friends):
        raise ValueError("User IDs must be non-negative integers for Spark ALS")
    if user > 2_147_483_647 or any(friend > 2_147_483_647 for friend in friends):
        raise ValueError("User IDs must be within signed 32-bit integer range for Spark ALS")
    return [(user, friends - {user})]


def to_canonical_pairs(row: tuple[int, frozenset[int]]) -> list[tuple[int, int]]:
    user, friends = row
    return [(min(user, friend), max(user, friend)) for friend in friends if friend != user]


def parse_edge_line(line: str) -> list[tuple[int, int]]:
    """Parse whitespace-separated source/target edges, skipping SNAP comments."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return []
    parts = stripped.split()
    if len(parts) < 2:
        raise ValueError(f"Expected two IDs: {line[:120]!r}")
    source, target = int(parts[0]), int(parts[1])
    if source < 0 or target < 0:
        raise ValueError("Negative node IDs are unsupported")
    return [(source, target)]
