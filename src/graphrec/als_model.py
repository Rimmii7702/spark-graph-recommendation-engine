"""Spark MLlib ALS collaborator filtering on binary friendship interactions.

Positive friendships are split as undirected pairs BEFORE direction expansion.
Sampled negatives (not observed dislikes) are screened against ALL known positive
links so withheld friendships are never mislabeled as non-friends.
"""
from __future__ import annotations

import random
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.sql.window import Window


INTERACTION_SCHEMA = StructType([
    StructField("user", IntegerType(), False),
    StructField("item", IntegerType(), False),
])


def to_directed(pairs):
    return pairs.select(F.col("u").cast("int").alias("user"), F.col("v").cast("int").alias("item")).unionByName(
        pairs.select(F.col("v").cast("int").alias("user"), F.col("u").cast("int").alias("item"))
    )


def negative_candidates(pair_iterator, user_pool, seed):
    """Deterministically propose two negatives per positive undirected edge."""
    if not user_pool:
        return
    n = len(user_pool)
    for u, v in pair_iterator:
        rng = random.Random((u * 1000003 + v * 9176 + seed) & ((1 << 63) - 1))
        c, d = user_pool[rng.randrange(n)], user_pool[rng.randrange(n)]
        if c != u:
            yield (int(u), int(c))
        if d != v:
            yield (int(v), int(d))


def fit_and_recommend(spark, all_pairs, train_pairs, test_pairs, vertices_df,
                      user_pool: list[int], targets: list[int], *, k=10,
                      seed=42, tune=False, rank=10, reg=0.1, max_iter=8):
    positives = to_directed(all_pairs).cache()
    positives_train = to_directed(train_pairs).withColumn("rating", F.lit(1.0))
    positives_test = to_directed(test_pairs).withColumn("rating", F.lit(1.0))
    users_broadcast = spark.sparkContext.broadcast(sorted(set(user_pool)))
    try:
        # Generate training and test negatives from their OWN positive split.
        # Test-edge endpoints must not drive training negative generation.
        negative_train_raw = spark.createDataFrame(
            train_pairs.rdd.mapPartitions(
                lambda rows: negative_candidates(rows, users_broadcast.value, seed)
            ), INTERACTION_SCHEMA).dropDuplicates()
        negative_test_raw = spark.createDataFrame(
            test_pairs.rdd.mapPartitions(
                lambda rows: negative_candidates(rows, users_broadcast.value, seed + 1)
            ), INTERACTION_SCHEMA).dropDuplicates()
        negative_train = (negative_train_raw
                          .join(positives, ["user", "item"], "left_anti")
                          .withColumn("rating", F.lit(0.0))).cache()
        negative_test = (negative_test_raw
                         .join(positives, ["user", "item"], "left_anti")
                         .join(negative_train.select("user", "item"), ["user", "item"], "left_anti")
                         .withColumn("rating", F.lit(0.0))).cache()
        train = positives_train.unionByName(negative_train).cache()
        test = positives_test.unionByName(negative_test).cache()
        train_count, test_count = train.count(), test.count()
        if train_count < 5 or test_count < 1:
            raise ValueError("Insufficient training/test rows; use a larger input graph")
        try:
            als = ALS(userCol="user", itemCol="item", ratingCol="rating",
                      coldStartStrategy="drop", nonnegative=True,
                      rank=rank, regParam=reg, maxIter=max_iter, seed=seed)
            pipeline = Pipeline(stages=[als])
            metric = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
            if tune:
                grid = (ParamGridBuilder().addGrid(als.rank, [8, 16])
                        .addGrid(als.regParam, [0.05, 0.1])
                        .addGrid(als.maxIter, [5, 10]).build())
                cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid,
                                    evaluator=metric, numFolds=2, parallelism=2, seed=seed)
                fitted = cv.fit(train)
                best = fitted.bestModel
                winning = min(range(len(fitted.avgMetrics)), key=lambda idx: fitted.avgMetrics[idx])
                chosen = {"rank": grid[winning][als.rank],
                          "reg_param": grid[winning][als.regParam],
                          "max_iter": grid[winning][als.maxIter]}
            else:
                best = pipeline.fit(train)
                chosen = {"rank": rank, "reg_param": reg, "max_iter": max_iter}
            model = best.stages[-1]
            scored_test = model.transform(test).cache()
            scored_count = scored_test.count()
            rmse = round(float(metric.evaluate(scored_test)), 6) if scored_count else None
            scored_test.unpersist()

            # Score candidate users not observed as friends in TRAIN.
            # Crucially, DO NOT exclude held-out positives from this candidate set.
            target_df = spark.createDataFrame([(int(x),) for x in targets], "user INT")
            item_df = vertices_df.select(F.col("id").cast("int").alias("item"))
            possibilities = target_df.crossJoin(item_df).filter(F.col("user") != F.col("item"))
            possibilities = possibilities.join(to_directed(train_pairs), ["user", "item"], "left_anti")
            scores = model.transform(possibilities)
            order = Window.partitionBy("user").orderBy(F.desc("prediction"), F.asc("item"))
            top = (scores.withColumn("position", F.row_number().over(order))
                   .where(F.col("position") <= k)
                   .orderBy("user", "position")
                   .select("user", "item", "prediction")
                   .collect())
            output = {user: [] for user in targets}
            for row in top:
                output[row["user"]].append(row["item"])
            stats = {
                "train_binary_interactions": train_count,
                "test_binary_interactions": test_count,
                "scored_test_interactions": scored_count,
                "test_score_coverage": round(scored_count / test_count, 6),
                "test_sampled_binary_rmse": rmse,
                "hyperparameters": chosen,
                "cross_validation": bool(tune),
                "negative_pool_size": len(user_pool),
                "note": "RMSE uses observed positives and sampled nonedges; ranking metrics use held-out positive edges."
            }
            return output, stats
        finally:
            train.unpersist()
            test.unpersist()
            negative_train.unpersist()
            negative_test.unpersist()
    finally:
        users_broadcast.unpersist()
        positives.unpersist()
