# Measuring recommendation quality and performance

## Ranking

Hold out 20% of unique canonical undirected friendship pairs *before* expanding reciprocals. For each evaluated user `u`, let `H_u` be their held-out true friends and `R_u(k)` be the first `k` recommendations generated **without training on those edges**.

| Metric | Formula | Interpretation |
|---|---|---|
| Precision@k | Mean across users of `len(R_u(k) ∩ H_u) / k` | Fraction of available recommendation slots containing withheld true links |
| Recall@k | Mean across users of `len(R_u(k) ∩ H_u) / len(H_u)` | Fraction of the user's withheld edges recovered |
| Hit Rate@k | Fraction of evaluated users with `len(R_u(k) ∩ H_u) >= 1` | How often at least one withheld link is recovered |
| Coverage | Fraction of evaluated users receiving at least one recommendation | Ability to return recommendations |
| Candidate overlap | Mean size of the intersection between method A and B's top-k lists | Similarity of recommendations, **not model quality** |

The script samples evaluation users only from users with >=1 held-out edge. Metrics are explicitly conditional on this selected group, **not** a population-wide accuracy number. The split is stochastic but seeded; results may change with Spark version and partitioning.

## ALS RMSE

The explicit-feedback ALS baseline uses **observed positive friendships (rating=1)** and sampled non-edges **(assumed rating=0)**. It reports `RMSE = sqrt(mean((y - prediction)^2))` on scored held-out positive/sampled-negative interactions.

- False-negative risks are reduced by screening negative proposals against *all* true edges.
- `coldStartStrategy='drop'` means some test interactions have no prediction. The output reports `scored_test_interactions / test_binary_interactions` so RMSE is not misinterpreted as applying to all test examples.
- RMSE depends on how negatives are sampled and does **not** equal classification accuracy or Precision@10.
- This is an offline simulation of link recovery. There is no real product event stream or feedback from actual users.

## Performance and hardware

`metrics/recommendations.json` records elapsed time for each recommendation method and the overall run. `metrics/wiki_vote_graph.json` records load, degree, PageRank, component, and triangle runtimes. Both measure wall-clock time on the machine running Spark. See the [benchmark provenance](results.md) for the hardware, configuration, run command, and verification details.
