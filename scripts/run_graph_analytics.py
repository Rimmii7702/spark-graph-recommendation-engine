"""GraphFrames analytics for the *directed* SNAP Wikipedia voting graph."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphrec.graph_analytics import analyze_network


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/wiki-Vote.txt.gz")
    p.add_argument("--output", type=Path, default=Path("metrics/wiki_vote_graph.json"))
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--shuffle-partitions", type=int, default=8)
    a = p.parse_args()
    if not Path(a.input).exists():
        raise SystemExit("Graph input missing; run python scripts/download_wiki_vote.py")
    if a.iterations < 1:
        p.error("iterations must be >= 1")
    from pyspark.sql import SparkSession

    start = perf_counter()
    spark = (SparkSession.builder.appName("GraphRec-Wikipedia-Network-Analytics")
             .config("spark.sql.shuffle.partitions", a.shuffle_partitions)
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    try:
        record = {
            "dataset": a.input,
            **analyze_network(spark, a.input, a.iterations),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "total_runtime_seconds": round(perf_counter() - start, 3),
        }
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record, indent=2))
        print(f"Saved: {a.output.resolve()}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
