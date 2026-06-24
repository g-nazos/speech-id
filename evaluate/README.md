# Evaluate Package

This package contains the speaker identification evaluation code for VoxCeleb1.
It compares multiple search strategies, measures accuracy and search time, and
writes a JSON summary to `results/speaker_search_evaluation.json` by default.

## Package Layout

- `evaluate_search.py` - script entrypoint for running the evaluator.
- `runner.py` - CLI orchestration and experiment loop.
- `config.py` - shared paths, environment loading, and connection string helpers.
- `data.py` - test embedding loading and database data access helpers.
- `search.py` - routing and ranking logic for the search strategies.
- `metrics.py` - accuracy, Hit@K, precision, recall, F1, and EER calculations.
- `reporting.py` - Markdown and JSON report output.
- `__init__.py` - package marker and convenience import for `main`.

## Experiments

The evaluator runs a grid of speaker search experiments over the unseen test
embeddings in `data/voxceleb_test_embeddings.pt`.

It compares three architectures:

- `brute_force` - search the full embedding database.
- `metadata_hierarchy` - route by `gender` and then by `gender:nationality` before searching a filtered subset.
- `kmeans_clusters` - route the query to one or more nearest cluster centroids, then search only within those clusters.

It also compares aggregation modes:

- `speaker_centroid` - average embeddings per speaker and rank speaker centroids directly.
- `closest` - search individual utterances and keep the closest speaker match.
- `vote` - search individual utterances and aggregate the top hits by voting.

Distance metrics can be configured from the CLI. The default is cosine similarity,
with Euclidean distance also supported.

## Running It

From the project root:

```bash
python evaluate/evaluate_search.py
```

Useful options:

- `--max-queries` to limit the number of test samples.
- `--metrics cosine euclidean` to evaluate multiple distances.
- `--cluster-probes 1 4 8` to test different cluster routing depths.
- `--utterance-mode all|closest|vote` to control which direct-utterance variants are evaluated.
- `--output-file path/to/result.json` to change the JSON output location.

## Output

The script prints a Markdown-style summary to the terminal and saves a JSON file
with the same experiment results. The default JSON output is:

`results/speaker_search_evaluation.json`
