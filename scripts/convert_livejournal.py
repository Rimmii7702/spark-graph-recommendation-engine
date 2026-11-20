"""Optional: convert a SNAP directed edge list into an UNDIRECTED adjacency-list dataset.

WARNING: The result is a symmetrized graph and may differ from the original
LiveJournal adjacency dataset. This is a conversion, not an exact reproduction.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrec.data import parse_edge_line


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Path to SNAP edge list (txt or gz)")
    p.add_argument("--output", default="data/converted_livejournal_adj")
    a = p.parse_args()
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("GraphRec-EdgeListToAdjacency").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        raw = spark.sparkContext.textFile(a.input)
        edges = (raw.flatMap(parse_edge_line)
                 .filter(lambda uv: uv[0] != uv[1])
                 .map(lambda uv: (min(uv), max(uv))).distinct())
        neighbors = (edges.flatMap(lambda uv: [uv, (uv[1], uv[0])])
                     .groupByKey().mapValues(lambda vs: ",".join(map(str, sorted(set(vs))))))
        # Spark creates a folder of part-000.. files: do not pre-create it.
        neighbors.sortByKey().map(lambda kv: f"{kv[0]}\t{kv[1]}").saveAsTextFile(a.output)
        print(f"Saved adjacency part files under {a.output}")
        print("Input is directed; output was symmetrized into UNDIRECTED friendships.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
