# Dataset setup

## LiveJournal recommendation benchmark

Download the undirected benchmark input from the project root.

```bash
curl --fail --location https://an-ml.s3.us-west-1.amazonaws.com/soc-LiveJournal1Adj.txt --output data/soc-LiveJournal1Adj.txt
```

On Windows PowerShell, use `curl.exe` to call curl directly. The [benchmark record](../results/livejournal_recommendations.json) contains the reference SHA-256 under `verification.input_validation.sha256`.

Each row uses `user_id<TAB>friend_id,friend_id,...`. The format allows an empty neighbor list. The parser removes self-links, and the pipeline canonicalizes each edge and deduplicates the result. The checked-in `sample_livejournal_adj.txt` fixture is synthetic and is intended for smoke tests.

The public [Stanford SNAP LiveJournal network](https://snap.stanford.edu/data/soc-LiveJournal1.html) uses directed edge pairs and represents a separate input. To benchmark a symmetrized version, download `soc-LiveJournal1.txt.gz` from SNAP, place it in `data/`, and run the converter.

```bash
docker compose run --rm spark spark-submit --master 'local[4]' --driver-memory 4g scripts/convert_livejournal.py --input data/soc-LiveJournal1.txt.gz --output data/converted_livejournal_adj
```

Use `--input data/converted_livejournal_adj` with the recommendation entry point and label the results as a symmetrized SNAP run.

## Wikipedia voting graph

Run `python scripts/download_wiki_vote.py` from the project root to download the original `data/wiki-Vote.txt.gz` from [Stanford SNAP](https://snap.stanford.edu/data/wiki-Vote.html). Each directed edge represents a vote from one user to another. Both the Spark pipeline and the verifier accept the compressed file directly.
