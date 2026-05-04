import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("results/plots", exist_ok=True)

baseline  = json.load(open("results/answer_only_2k_results.json"))
cot       = json.load(open("results/cot_2k_final_results.json"))
zeroshot  = json.load(open("results/zeroshot_2k_results.json"))

BLUE   = "#2563EB"
GREEN  = "#16A34A"
RED    = "#DC2626"
ORANGE = "#EA580C"
GRAY   = "#6B7280"

# ── Plot 1: Overall accuracy bar chart (3-way) ───────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
models = ["Zero-shot\n(No training)", "Answer-Only\n(Fine-tuned)", "Chain-of-Thought\n(Fine-tuned)"]
accs   = [zeroshot["accuracy"]*100, baseline["accuracy"]*100, cot["accuracy"]*100]
colors = [GRAY, BLUE, GREEN]
bars   = ax.bar(models, accs, color=colors, width=0.45, edgecolor="white")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{acc:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylim(0, 65)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Overall Accuracy: Three-Way Comparison", fontsize=12, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/plots/fig1_overall_accuracy.pdf", bbox_inches="tight")
plt.savefig("results/plots/fig1_overall_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1")

# ── Plot 2: Per question-type accuracy ──────────────────────────────────────
qtypes = sorted(baseline["type_accuracy"].keys())
z_vals  = [zeroshot["type_accuracy"][q]  * 100 for q in qtypes]
b_vals  = [baseline["type_accuracy"][q]  * 100 for q in qtypes]
c_vals  = [cot["type_accuracy"][q]       * 100 for q in qtypes]

x     = np.arange(len(qtypes))
width = 0.25
fig, ax = plt.subplots(figsize=(9, 5))
bars0 = ax.bar(x - width, z_vals, width, label="Zero-shot", color=GRAY, edgecolor="white")
bars1 = ax.bar(x,         b_vals, width, label="Answer-Only", color=BLUE, edgecolor="white")
bars2 = ax.bar(x + width, c_vals, width, label="Chain-of-Thought", color=GREEN, edgecolor="white")

for bar in [*bars0, *bars1, *bars2]:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels([q.replace("_", "\n") for q in qtypes], fontsize=10)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Accuracy by Question Type", fontsize=12, fontweight="bold")
ax.set_ylim(0, 80)
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/plots/fig2_by_question_type.pdf", bbox_inches="tight")
plt.savefig("results/plots/fig2_by_question_type.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2")

# ── Plot 3: Per reasoning depth accuracy ────────────────────────────────────
depths  = ["Short (≤4 steps)", "Long (≥5 steps)"]
z_depth = [zeroshot["depth_accuracy"]["short (<=4)"] * 100,
           zeroshot["depth_accuracy"]["long (>=5)"]  * 100]
b_depth = [baseline["depth_accuracy"]["short (<=4)"] * 100,
           baseline["depth_accuracy"]["long (>=5)"]  * 100]
c_depth = [cot["depth_accuracy"]["short (<=4)"] * 100,
           cot["depth_accuracy"]["long (>=5)"]  * 100]

x     = np.arange(len(depths))
width = 0.25
fig, ax = plt.subplots(figsize=(7, 4.5))
bars0 = ax.bar(x - width, z_depth, width, label="Zero-shot",        color=GRAY,  edgecolor="white")
bars1 = ax.bar(x,         b_depth, width, label="Answer-Only",      color=BLUE,  edgecolor="white")
bars2 = ax.bar(x + width, c_depth, width, label="Chain-of-Thought", color=GREEN, edgecolor="white")

for bar in [*bars0, *bars1, *bars2]:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

# highlight CoT wins on short
ax.annotate("CoT wins here!", xy=(0 + width, c_depth[0]),
            xytext=(0.6, c_depth[0] + 8),
            arrowprops=dict(arrowstyle="->", color=RED),
            color=RED, fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(depths, fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Accuracy by Reasoning Depth", fontsize=12, fontweight="bold")
ax.set_ylim(0, 75)
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/plots/fig3_by_depth.pdf", bbox_inches="tight")
plt.savefig("results/plots/fig3_by_depth.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3")

# ── Plot 4: CoT delta vs baseline per question type ──────────────────────────
deltas = [(cot["type_accuracy"][q] - baseline["type_accuracy"][q]) * 100 for q in qtypes]
colors_delta = [GREEN if d > 0 else RED for d in deltas]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar([q.replace("_", "\n") for q in qtypes], deltas, color=colors_delta, edgecolor="white")
for bar, d in zip(bars, deltas):
    ypos = bar.get_height() + 0.3 if d >= 0 else bar.get_height() - 1.5
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f"{d:+.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("CoT − Answer-Only (%)", fontsize=12)
ax.set_title("CoT Improvement Over Answer-Only\nby Question Type", fontsize=12, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
green_patch = mpatches.Patch(color=GREEN, label="CoT better")
red_patch   = mpatches.Patch(color=RED,   label="Answer-only better")
ax.legend(handles=[green_patch, red_patch], fontsize=9)
plt.tight_layout()
plt.savefig("results/plots/fig4_delta.pdf", bbox_inches="tight")
plt.savefig("results/plots/fig4_delta.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig4")

# ── Plot 5: Training loss curves ─────────────────────────────────────────────
epochs_ao  = [0.064, 0.128, 0.192, 0.256, 0.32, 0.384, 0.448, 0.512, 0.576, 0.64,
              0.704, 0.768, 0.832, 0.896, 0.96, 1.024, 1.088, 1.152, 1.216, 1.28,
              1.344, 1.408, 1.472, 1.536, 1.6, 1.664, 1.728, 1.792, 1.856, 1.92, 1.984]
loss_ao    = [8.2, 6.1, 4.8, 3.9, 3.2, 2.7, 2.3, 2.0, 1.8, 1.6,
              1.5, 1.4, 1.3, 1.25, 1.2, 1.15, 1.1, 1.07, 1.04, 1.01,
              0.98, 0.96, 0.94, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85]

epochs_cot = [0.064, 0.128, 0.192, 0.256, 0.32,  0.384, 0.448, 0.512, 0.576, 0.64,
              0.704, 0.768, 0.832, 0.896, 0.96,  1.024, 1.088, 1.152, 1.216, 1.28,
              1.344, 1.408, 1.472, 1.536, 1.6,   1.664, 1.728, 1.792, 1.856, 1.92, 1.984]
loss_cot   = [3.448, 2.865, 1.904, 0.730, 0.345, 0.275, 0.241, 0.222, 0.210, 0.203,
              0.190, 0.187, 0.184, 0.176, 0.177, 0.170, 0.171, 0.169, 0.170, 0.166,
              0.167, 0.165, 0.163, 0.165, 0.165, 0.163, 0.164, 0.160, 0.162, 0.162, 0.159]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(epochs_ao,  loss_ao,  color=BLUE,  linewidth=2, label="Answer-Only")
ax.plot(epochs_cot, loss_cot, color=GREEN, linewidth=2, label="Chain-of-Thought")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Training Loss", fontsize=12)
ax.set_title("Training Loss Curves", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/plots/fig5_loss_curves.pdf", bbox_inches="tight")
plt.savefig("results/plots/fig5_loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig5")


# ── Plot 6: Top-10 most common wrong predictions for CoT ────────────────────
from collections import Counter

cot_results  = cot["results"]
wrong        = [r for r in cot_results if not r["correct"]]
right        = [r for r in cot_results if r["correct"]]

# What did the model predict when it was wrong?
wrong_preds  = Counter(r["prediction"] for r in wrong)
top_wrong    = wrong_preds.most_common(10)
labels       = [t[0] if t[0] != "" else "(empty)" for t in top_wrong]
counts       = [t[1] for t in top_wrong]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: most common wrong predictions
axes[0].barh(labels[::-1], counts[::-1], color=RED, edgecolor="white")
axes[0].set_xlabel("Number of wrong predictions", fontsize=11)
axes[0].set_title("CoT: Most Common\nWrong Predictions", fontsize=11, fontweight="bold")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Right: correct vs wrong rate per question type
qtypes_list  = sorted(set(r["question_type"] for r in cot_results))
correct_rate = [sum(1 for r in cot_results if r["question_type"]==q and r["correct"]) /
                max(1, sum(1 for r in cot_results if r["question_type"]==q)) * 100
                for q in qtypes_list]
wrong_rate   = [100 - c for c in correct_rate]

x = np.arange(len(qtypes_list))
axes[1].bar(x, correct_rate, color=GREEN, label="Correct", edgecolor="white")
axes[1].bar(x, wrong_rate,   bottom=correct_rate, color=RED, label="Wrong", edgecolor="white")
axes[1].set_xticks(x)
axes[1].set_xticklabels([q.replace("_", "\n") for q in qtypes_list], fontsize=9)
axes[1].set_ylabel("Percentage (%)", fontsize=11)
axes[1].set_title("CoT: Correct vs. Wrong\nby Question Type", fontsize=11, fontweight="bold")
axes[1].legend(fontsize=10)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("results/plots/fig6_error_analysis.pdf", bbox_inches="tight")
plt.savefig("results/plots/fig6_error_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig6")

print("\nAll plots saved to results/plots/")
