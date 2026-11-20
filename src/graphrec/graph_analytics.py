"""Spark GraphFrames analytics for a directed voting network."""
from __future__ import annotations

from time import perf_counter


def top_five(df, name):
    return [row.asDict() for row in df.orderBy(df[name].desc(), df.id.asc()).limit(5).collect()]


def analyze_network(spark, input_path: str, iterations: int) -> dict:
    from graphframes import GraphFrame
    from pyspark import StorageLevel
    from pyspark.sql import functions as F

    durations = {}
    tic = perf_counter()
    cleaned = (spark.read.text(input_path)
               .select(F.trim("value").alias("line"))
               .filter((F.col("line") != "") & ~F.col("line").startswith("#"))
               .select(F.split(F.regexp_replace("line", r"\s+", " "), " ").alias("parts"))
               .where(F.size("parts") >= 2))
    edges = (cleaned.select(F.col("parts")[0].cast("long").alias("src"),
                            F.col("parts")[1].cast("long").alias("dst"))
             .na.drop().dropDuplicates(["src", "dst"]).cache())
    vertices = (edges.select(F.col("src").alias("id"))
                .union(edges.select(F.col("dst").alias("id"))).distinct().cache())
    node_count, edge_count = vertices.count(), edges.count()
    durations["load_seconds"] = round(perf_counter() - tic, 3)
    graph = GraphFrame(vertices, edges)
    tic = perf_counter()
    out = top_five(graph.outDegrees, "outDegree")
    ins = top_five(graph.inDegrees, "inDegree")
    durations["degree_seconds"] = round(perf_counter() - tic, 3)
    tic = perf_counter()
    ranks = graph.pageRank(resetProbability=0.15, maxIter=iterations)
    top_pr = top_five(ranks.vertices.select("id", "pagerank"), "pagerank")
    ranks.unpersist()
    durations["pagerank_seconds"] = round(perf_counter() - tic, 3)
    tic = perf_counter()
    cc = graph.connectedComponents(algorithm="two_phase", useLabelsAsComponents=True,
                                    use_local_checkpoints=True)
    components = cc.groupBy("component").count()
    largest_components = [r.asDict() for r in
                          components.orderBy(F.desc("count"), F.asc("component")).limit(5).collect()]
    cc.unpersist()
    durations["components_seconds"] = round(perf_counter() - tic, 3)
    tic = perf_counter()
    triangles = graph.triangleCount(storage_level=StorageLevel.MEMORY_AND_DISK)
    top_tri = top_five(triangles.select("id", "count"), "count")
    triangles.unpersist()
    durations["triangle_count_seconds"] = round(perf_counter() - tic, 3)
    vertices.unpersist()
    edges.unpersist()
    return {
        "network_semantics": "Directed source-to-target votes; triangle topology ignores direction",
        "vertex_count": node_count,
        "edge_count": edge_count,
        "pagerank_iterations": iterations,
        "a_outdegree": out,
        "b_indegree": ins,
        "c_pagerank": top_pr,
        "d_components": largest_components,
        "e_triangles": top_tri,
        "runtimes_seconds": durations,
    }
