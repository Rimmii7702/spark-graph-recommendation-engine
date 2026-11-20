"""Two-hop mutual-friend recommendation implemented using Spark RDD MapReduce."""
from __future__ import annotations


def recommend_mutual(spark_context, train_pairs_rdd, target_users: list[int], k: int = 10) -> dict[int, list[int]]:
    if not target_users:
        return {}
    targets = sorted(set(target_users))
    target_set = spark_context.broadcast(set(targets))
    # One canonical edge (a,b) becomes reciprocal adjacency entries.
    adjacency = (train_pairs_rdd
                 .flatMap(lambda edge: [edge, (edge[1], edge[0])])
                 .groupByKey()
                 .mapValues(lambda neighbors: set(neighbors))
                 .cache())
    try:
        target_neighbors = dict(adjacency.filter(lambda item: item[0] in target_set.value).collect())
        for user in targets:
            target_neighbors.setdefault(user, set())
        known = spark_context.broadcast(target_neighbors)
        try:
            # Join shared intermediary -> its neighbors, then count two-hop paths.
            through = [(friend, user) for user, friends in target_neighbors.items() for friend in friends]
            if not through:
                return {user: [] for user in targets}
            candidates = (spark_context.parallelize(through)
                          .join(adjacency)
                          .flatMap(lambda item: [((item[1][0], candidate), 1)
                                                 for candidate in item[1][1]]))
            counts = candidates.reduceByKey(lambda x, y: x + y)
            eligible = counts.filter(
                lambda item: item[0][0] != item[0][1]
                and item[0][1] not in known.value.get(item[0][0], set())
            )
            grouped = eligible.map(lambda item: (item[0][0], (item[0][1], item[1]))).groupByKey()
            top = grouped.mapValues(
                lambda pairs: [candidate for candidate, count in
                               sorted(pairs, key=lambda pair: (-pair[1], pair[0]))[:k]]
            ).collectAsMap()
            return {user: list(top.get(user, [])) for user in targets}
        finally:
            known.unpersist()
    finally:
        target_set.unpersist()
        adjacency.unpersist()
