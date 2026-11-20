"""Evaluate mutual-friend MapReduce and ALS on one held-out LiveJournal split."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrec.data import parse_adjacency, to_canonical_pairs
from graphrec.metrics import compare_recommendations, evaluate_ranking
from graphrec.mutual import recommend_mutual


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default="data/sample_livejournal_adj.txt")
    parser.add_argument("--output", type=Path, default=Path("metrics/recommendations.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--users", type=int, default=10, help="Number of evaluation users with held-out links")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--method", choices=["both", "mutual", "als"], default="both")
    parser.add_argument("--negative-pool", type=int, default=20000)
    parser.add_argument("--tune-als", action="store_true", help="2-fold CV over 8 ALS configurations; expensive")
    parser.add_argument("--shuffle-partitions", type=int, default=8)
    args = parser.parse_args()
    if not 0 < args.test_fraction < 1 or min(args.users, args.top_k, args.negative_pool) < 1:
        parser.error("Use test-fraction within (0,1) and positive users, top-k and negative-pool")
    return args


def main():
    from pyspark import StorageLevel
    from pyspark.sql import SparkSession, functions as F
    from pyspark.sql.types import StructType, StructField, IntegerType

    a = arguments()
    if not Path(a.input).exists():
        raise SystemExit(f"Input {a.input} not found. Read data/README.md")
    started = perf_counter()
    spark = (SparkSession.builder.appName("GraphRec-FriendRecommendations")
             .config("spark.sql.shuffle.partitions", a.shuffle_partitions)
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    sc = spark.sparkContext
    try:
        rows = sc.textFile(a.input).flatMap(parse_adjacency).persist(StorageLevel.MEMORY_AND_DISK)
        canonical = rows.flatMap(to_canonical_pairs).distinct().persist(StorageLevel.MEMORY_AND_DISK)
        vertices = (rows.flatMap(lambda row: [row[0], *row[1]])
                    .distinct().persist(StorageLevel.MEMORY_AND_DISK))
        n_edges, n_users = canonical.count(), vertices.count()
        if n_edges < 40:
            raise ValueError("At least 40 undirected links are recommended for a useful split")
        schema = StructType([StructField("u", IntegerType(), False),
                             StructField("v", IntegerType(), False)])
        pairs = spark.createDataFrame(canonical, schema).cache()
        pairs.count()
        train, test = pairs.randomSplit([1 - a.test_fraction, a.test_fraction], seed=a.seed)
        train, test = train.cache(), test.cache()
        training_count, testing_count = train.count(), test.count()
        if testing_count < 1:
            raise ValueError("Test split contains no positive edges; increase dataset size")
        eligible_users = test.rdd.flatMap(lambda row: (row.u, row.v)).distinct().cache()
        eligible_count = eligible_users.count()
        targets = sorted(eligible_users.takeSample(False, min(a.users, eligible_count), a.seed))
        selected = set(targets)
        known = sc.broadcast(selected)
        try:
            truth_pairs = (test.rdd.flatMap(lambda row: [(row.u, row.v), (row.v, row.u)])
                           .filter(lambda uv: uv[0] in known.value)
                           .collect())
        finally:
            known.unpersist()
        truth = {u: set() for u in targets}
        for user, friend in truth_pairs:
            truth[user].add(friend)
        summary = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "input": a.input,
            "dataset_kind": "synthetic demo" if Path(a.input).name.startswith("sample_") else "user-supplied adjacency",
            "method": a.method,
            "random_seed": a.seed,
            "top_k": a.top_k,
            "test_fraction": a.test_fraction,
            "vertex_count": n_users,
            "undirected_edge_count": n_edges,
            "train_undirected_edges": training_count,
            "held_out_undirected_edges": testing_count,
            "selected_evaluation_users": targets,
            "evaluation_protocol": "Canonical undirected edge split before reverse directions; held-out friendships available as recommendations",
            "results": {},
        }
        try:
            if a.method in ("both", "mutual"):
                start = perf_counter()
                m = recommend_mutual(sc, train.rdd.map(lambda row: (row.u, row.v)), targets, a.top_k)
                summary["results"]["mutual_friends"] = {
                    "metrics": evaluate_ranking(m, truth, a.top_k),
                    "runtime_seconds": round(perf_counter() - start, 3),
                    "top_k_recommendations": {str(user): m[user] for user in targets},
                }
            if a.method in ("both", "als"):
                from graphrec.als_model import fit_and_recommend
                start = perf_counter()
                pool = vertices.takeSample(False, min(n_users, a.negative_pool), a.seed + 1)
                vertices_df = (pairs.select(F.col("u").alias("id"))
                               .union(pairs.select(F.col("v").alias("id"))).distinct())
                als_recs, als_stats = fit_and_recommend(
                    spark, pairs, train, test, vertices_df, pool, targets,
                    k=a.top_k, seed=a.seed, tune=a.tune_als)
                summary["results"]["als"] = {
                    "metrics": evaluate_ranking(als_recs, truth, a.top_k),
                    "binary_interaction_evaluation": als_stats,
                    "runtime_seconds": round(perf_counter() - start, 3),
                    "top_k_recommendations": {str(user): als_recs[user] for user in targets},
                }
            if a.method == "both":
                summary["method_comparison"] = compare_recommendations(m, als_recs, a.top_k)
        finally:
            eligible_users.unpersist()
            train.unpersist()
            test.unpersist()
            pairs.unpersist()
            vertices.unpersist()
            canonical.unpersist()
            rows.unpersist()
        summary["total_runtime_seconds"] = round(perf_counter() - started, 3)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k not in ("results",)}, indent=2))
        for name, detail in summary["results"].items():
            print(f"\n{name}: {json.dumps(detail['metrics'], indent=2)}")
        print(f"\nSaved: {a.output.resolve()}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
