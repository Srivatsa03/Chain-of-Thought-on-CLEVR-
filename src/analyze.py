"""
Reads results/answer_only_results.json and results/chain_of_thought_results.json
and produces all comparison plots + a summary table.
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from collections import defaultdict

matplotlib.use("Agg")

RESULTS = {
    "answer_only":      "results/answer_only_results.json",
    "chain_of_thought": "results/chain_of_thought_results.json",
}
OUT_DIR = "results"


def load(path):
    with open(path) as f:
        return json.load(f)


def plot_overall(baseline, cot):
    labels = ["Baseline\n(Answer-Only)", "CoT Model\n(Chain-of-Thought)"]
    accs   = [baseline["accuracy"], cot["accuracy"]]
    colors = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, accs, color=colors, width=0.4, edgecolor="black")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Overall VQA Accuracy on CLEVR", fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "overall_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_by_type(baseline, cot):
    types_b = baseline["type_accuracy"]
    types_c = cot["type_accuracy"]
    all_types = sorted(set(types_b) | set(types_c))

    b_vals = [types_b.get(t, 0) for t in all_types]
    c_vals = [types_c.get(t, 0) for t in all_types]

    x = np.arange(len(all_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, b_vals, width, label="Baseline", color="#4C72B0", edgecolor="black")
    ax.bar(x + width / 2, c_vals, width, label="CoT Model", color="#DD8452", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(all_types, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy by Question Type", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "accuracy_by_type.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_by_depth(baseline, cot):
    depths_b = baseline["depth_accuracy"]
    depths_c = cot["depth_accuracy"]
    all_buckets = sorted(set(depths_b) | set(depths_c))

    b_vals = [depths_b.get(k, 0) for k in all_buckets]
    c_vals = [depths_c.get(k, 0) for k in all_buckets]

    x = np.arange(len(all_buckets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, b_vals, width, label="Baseline", color="#4C72B0", edgecolor="black")
    ax.bar(x + width / 2, c_vals, width, label="CoT Model", color="#DD8452", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(all_buckets, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy by Reasoning Depth", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "accuracy_by_depth.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def print_summary_table(baseline, cot):
    print("\n" + "=" * 60)
    print(f"{'Metric':<30} {'Baseline':>12} {'CoT Model':>12} {'Delta':>8}")
    print("=" * 60)

    def row(label, b, c):
        delta = c - b
        sign  = "+" if delta >= 0 else ""
        print(f"{label:<30} {b:>12.4f} {c:>12.4f} {sign}{delta:>7.4f}")

    row("Overall Accuracy", baseline["accuracy"], cot["accuracy"])

    all_types = sorted(set(baseline["type_accuracy"]) | set(cot["type_accuracy"]))
    for t in all_types:
        b = baseline["type_accuracy"].get(t, 0)
        c = cot["type_accuracy"].get(t, 0)
        row(f"  {t}", b, c)

    all_depths = sorted(set(baseline["depth_accuracy"]) | set(cot["depth_accuracy"]))
    for d in all_depths:
        b = baseline["depth_accuracy"].get(d, 0)
        c = cot["depth_accuracy"].get(d, 0)
        row(f"  {d}", b, c)

    print("=" * 60)

    # Save as CSV
    rows = []
    rows.append({"metric": "overall", "baseline": baseline["accuracy"], "cot": cot["accuracy"]})
    for t in all_types:
        rows.append({"metric": f"type_{t}",
                     "baseline": baseline["type_accuracy"].get(t, 0),
                     "cot":      cot["type_accuracy"].get(t, 0)})
    for d in all_depths:
        rows.append({"metric": f"depth_{d}",
                     "baseline": baseline["depth_accuracy"].get(d, 0),
                     "cot":      cot["depth_accuracy"].get(d, 0)})

    df = pd.DataFrame(rows)
    df["delta"] = df["cot"] - df["baseline"]
    csv_path = os.path.join(OUT_DIR, "summary_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary table saved to {csv_path}")


def qualitative_examples(cot_results, n_good=3, n_bad=3):
    results = cot_results["results"]
    correct_samples   = [r for r in results if r["correct"]][:n_good]
    incorrect_samples = [r for r in results if not r["correct"]][:n_bad]

    lines = ["# Qualitative Examples\n\n## Correct Predictions\n"]
    for r in correct_samples:
        lines.append(f"**Q**: {r['question']}")
        lines.append(f"**GT**: {r['ground_truth']} | **Pred**: {r['prediction']}")
        lines.append(f"**Generated**: {r['generated_text']}\n")

    lines.append("## Incorrect Predictions\n")
    for r in incorrect_samples:
        lines.append(f"**Q**: {r['question']}")
        lines.append(f"**GT**: {r['ground_truth']} | **Pred**: {r['prediction']}")
        lines.append(f"**Generated**: {r['generated_text']}\n")

    out_path = os.path.join(OUT_DIR, "qualitative_examples.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    print("Loading results...")
    baseline = load(RESULTS["answer_only"])
    cot      = load(RESULTS["chain_of_thought"])

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Generating plots...")
    plot_overall(baseline, cot)
    plot_by_type(baseline, cot)
    plot_by_depth(baseline, cot)

    print_summary_table(baseline, cot)
    qualitative_examples(cot)

    print("\nAll analysis complete. Check the results/ directory.")
