# System design and pipeline

## Data flow A: Friend recommendation

1. Read the LiveJournal undirected adjacency-list file using a Spark RDD.
2. Parse each user/friend list, remove self-links, create canonical `(min_id,max_id)` undirected edges, and de-duplicate.
3. Randomly split **canonical undirected edges** into 80% training and 20% held-out test; freeze seed 42. This happens before duplicating edges into two directed ALS interactions.
4. Select up to 10 evaluation users with at least one held-out neighbor.
5. Run two independent methods against the same training split:
   - **MapReduce**: two-hop neighbor joins, `reduceByKey` mutual-friend scores, filter training friends, rank.
   - **ALS**: positive + sampled negative interactions, fit latent factors, score candidate users, filter training friends, rank.
6. Compare both ranked lists with the same hidden test edges: Precision@10, Recall@10, Hit Rate@10, Coverage@10, and overlap. Record wall-clock elapsed times separately.
7. Optionally tune ALS through `CrossValidator` (2 folds, 8 hyperparameter combinations) and compute held-out binary RMSE with the scored-row fraction.

**Boundary:** A withheld positive edge must not be excluded from recommendation candidates. Filtering against the *entire original graph* would invalidate offline Precision/Recall because it would remove every actual hit before evaluation. Negative sampling does exclude true positives to avoid mislabeling friendships as rating 0.

**Scaling tradeoff:** The RDD two-hop expansion may be expensive around high-degree users. ALS candidate scoring currently cross-joins the evaluation users with all known vertices, which can be expensive for multi-million-vertex graphs; for large deployments use approximate retrieval / ANN candidate generation and a distributed Spark cluster.

## Data flow B: Wikipedia voting graph

1. Download public `wiki-Vote.txt.gz`; ignore SNAP comment lines and read whitespace-delimited directed edges.
2. Deduplicate edges without symmetrizing.
3. Build `GraphFrame(vertices=[id], edges=[src,dst])`.
4. Compute top in-degree, out-degree and PageRank on the **directed** graph.
5. Compute weak connected-component sizes (ignoring direction for connectivity) and triangle counts (ignoring direction for triangle topology).
6. Save JSON containing counts and elapsed time for each algorithm.
7. Verify with an independent stdlib algorithm, including the published SNAP baseline (7,115 nodes, 103,689 edges, largest WCC 7,066, 608,389 triangles).

## Isolation between datasets

LiveJournal friendship links have different semantics from Wikipedia voting links. The two are separate tasks in a common graph analytics toolkit; no cross-dataset recommendation model or data join is asserted.
