# Benchmark provenance and reproduction

## LiveJournal recommendations

The [audited benchmark](../results/livejournal_recommendations.json) records the November 20, 2026 run on the original CS 6350 LiveJournal adjacency input. The archived JSON is a byte-for-byte copy of the reviewed output. The [benchmark input source](https://an-ml.s3.us-west-1.amazonaws.com/soc-LiveJournal1Adj.txt) supplies the file. Its SHA-256 is `964b93f1e58cce5377c9003eea6be12d24883f58299dfec97ee05f070b1ebf2d` and its size is 4,156,181 bytes.

An independent parse counted 49,995 vertices, including 871 isolated users, and 661,596 reciprocal adjacency entries representing 330,798 unique undirected edges. Spark applied weights 0.8 and 0.2 with seed 42. The audit checked that the two splits were disjoint and together contained every source edge. Both methods evaluated the same 10 users sampled with seed 42 from the 36,014 users with at least one held-out edge.

ALS used a 20,000-vertex negative pool, rank 10, regularization 0.1, eight iterations, and nonnegative factors. The fit used 529,166 positive and 528,412 sampled-negative interactions. The test contained 132,430 positive and 132,101 sampled-negative interactions, of which 261,074 received predictions. The audit independently recomputed the ranking metrics from the held-out sets and RMSE from the squared residuals. A separate timed execution of the standard entry point matched the audited split counts, selected users, recommendations, ranking metrics, and reported RMSE.

The timed run used Windows 11 build 26200, an Intel Core i7-1185G7, Java 17.0.20.1, Python 3.13.14, PySpark 4.2.0, NumPy 2.5.3, and a 4 GiB Spark driver. Spark ran four local task threads, two input and canonical-edge partitions, and eight SQL shuffle partitions. The overall time includes Spark startup, parsing, splitting, and both models. The timers stop before JSON writing and Spark shutdown. Spark version and partition changes can affect seeded split membership, scores, and timings.

To repeat the recorded setup in PowerShell, activate Python 3.13.14, set `JAVA_HOME` to Java 17.0.20.1, and run from the repository root. Keep newly generated outputs under `metrics/`.

```powershell
python -m pip install pyspark==4.2.0 numpy==2.5.3
$env:PYTHONPATH = (Resolve-Path src).Path
$env:PYSPARK_PYTHON = (Get-Command python).Source
$env:PYTHONHASHSEED = '0'
$env:SPARK_LOCAL_IP = '127.0.0.1'
$env:PYSPARK_SUBMIT_ARGS = '--master local[4,3] --driver-memory 4g --conf spark.ui.showConsoleProgress=false --conf spark.python.worker.faulthandler.enabled=true --conf spark.python.authenticate.socketTimeout=180s pyspark-shell'
python -u scripts/run_recommendations.py --input data/soc-LiveJournal1Adj.txt --method both --users 10 --top-k 10 --seed 42 --test-fraction 0.2 --negative-pool 20000 --shuffle-partitions 8 --output metrics/recommendations.json
```

## Wikipedia voting network

The November 12, 2026 [archived GraphFrames output](../results/archived_wikivote_results.json) retains the original results with personal team identifiers and a machine-specific path removed. The [verification report](../results/archived_wikivote_verification.json) retains the source digest and check outcomes. The graph contains 7,115 vertices, 103,689 directed edges, 24 weakly connected components, and 608,389 undirected triangles. PageRank used 10 iterations and a 0.15 reset probability.

The independently implemented verifier recomputes degrees, weak components, triangle counts and PageRank from the original [Stanford SNAP file](https://snap.stanford.edu/data/wiki-Vote.html). It compares the output with GraphFrames and checks SNAP's published graph totals. The figures under `docs/figures/` render the archived PageRank and triangle outputs. Use `scripts/plot_results.py` to regenerate them.
