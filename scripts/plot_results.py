"""Render two honest, labeled charts from the verified archived graph results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_plot(rows, value_key, title, xlabel, output):
    labels = [str(x["id"]) for x in rows]
    values = [float(x[value_key]) for x in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.barh(labels[::-1], values[::-1], color="#3973B9")
    ax.set_title(title, fontweight="bold", pad=16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Anonymized vertex ID")
    ax.bar_label(bars, fmt="%.2f" if value_key == "pagerank" else "%d", padding=4)
    ax.spines[['right','top']].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("results/archived_wikivote_results.json"))
    p.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = p.parse_args()
    stats = json.loads(args.input.read_text(encoding="utf-8"))
    make_plot(stats['c_pagerank'], 'pagerank',
              'Top 5 Wikipedia Vote PageRank Scores', 'PageRank (10 iterations)',
              args.output_dir/'wiki_vote_pagerank.png')
    make_plot(stats['e_triangles'], 'count',
              'Top 5 Vertices by Undirected Triangle Count', 'Triangles per vertex',
              args.output_dir/'wiki_vote_triangles.png')


if __name__ == '__main__':
    main()
