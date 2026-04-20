"""
experiment_variants.py — Person 4
==================================
Runs the 4 core variant comparisons required by the project spec and
produces publication-quality comparison charts + a summary CSV.

Usage
-----
    python experiment_variants.py

Outputs (written to ./results/)
--------------------------------
  variant_comparison.png   — convergence curves for all variants
  inertia_boxplot.png      — score distribution for inertia comparison
  topology_boxplot.png     — score distribution for topology comparison
  variant_summary.csv      — mean / std / best over N runs per variant
"""

import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import csv

# ── path fix (run from repo root) ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")

from src.pso.fitness import FitnessEvaluator
from src.pso.variants import build_variant, VARIANTS

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

N_RUNS        = 30       # runs per variant (30 as required by spec)
NUM_PARTICLES = 30
MAX_ITER      = 100
STEPS         = 100      # simulation steps inside fitness

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Seeds (fixed for reproducibility — store with results as required)
RNG = np.random.default_rng(0)
ALL_SEEDS = RNG.integers(0, 100_000, size=N_RUNS).tolist()

# Variants to compare in the experiment
EXPERIMENT_VARIANTS = [
    "constant_inertia",
    "linear_inertia",
    "global_topology",
    "local_topology",
    "constriction",
]

COLORS = {
    "constant_inertia": "#e41a1c",
    "linear_inertia":   "#377eb8",
    "global_topology":  "#4daf4a",
    "local_topology":   "#984ea3",
    "constriction":     "#ff7f00",
}

LABELS = {
    "constant_inertia": "Constant Inertia (w=0.5)",
    "linear_inertia":   "Linear Inertia (0.9 to 0.4)",
    "global_topology":  "Global Topology",
    "local_topology":   "Local Topology (ring)",
    "constriction":     "Constriction Factor",
}

# ─────────────────────────────────────────────────────────────────────────────
# Run experiments
# ─────────────────────────────────────────────────────────────────────────────

def run_all_variants():
    evaluator = FitnessEvaluator(steps=STEPS)
    all_results = {}   # variant to list of result dicts

    for vname in EXPERIMENT_VARIANTS:
        print(f"\n{'─'*55}")
        print(f"  Variant: {LABELS[vname]}")
        print(f"{'─'*55}")
        runs = []
        for i, seed in enumerate(ALL_SEEDS):
            t0  = time.perf_counter()
            opt = build_variant(
                vname,
                fitness_evaluator=evaluator,
                seed=int(seed),
                num_particles=NUM_PARTICLES,
                max_iterations=MAX_ITER,
            )
            res = opt.run()
            elapsed = time.perf_counter() - t0
            runs.append({
                "seed":     seed,
                "score":    res["best_score"],
                "position": res["best_position"],
                "history":  res["history"],
                "runtime":  elapsed,
            })
            print(f"  Run {i+1:2d}/{N_RUNS}  seed={seed:6d}  "
                  f"score={res['best_score']:8.1f}  ({elapsed:.1f}s)")
        all_results[vname] = runs

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Save seeds
# ─────────────────────────────────────────────────────────────────────────────

def save_seeds():
    path = os.path.join(RESULTS_DIR, "seeds.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_index", "seed"])
        for i, s in enumerate(ALL_SEEDS):
            w.writerow([i + 1, s])
    print(f"\n[seeds] saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_summary(all_results):
    path = os.path.join(RESULTS_DIR, "variant_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "variant", "mean_score", "std_score",
            "best_score", "worst_score", "mean_runtime_s",
        ])
        for vname, runs in all_results.items():
            scores   = [r["score"]   for r in runs]
            runtimes = [r["runtime"] for r in runs]
            w.writerow([
                LABELS[vname],
                f"{np.mean(scores):.2f}",
                f"{np.std(scores):.2f}",
                f"{np.min(scores):.2f}",
                f"{np.max(scores):.2f}",
                f"{np.mean(runtimes):.3f}",
            ])
    print(f"[summary] saved to {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Plot: convergence curves (mean ± std band)
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(all_results):
    fig, ax = plt.subplots(figsize=(10, 6))

    for vname, runs in all_results.items():
        # stack histories to shape (N_RUNS, iterations+1)
        histories = np.array([r["history"] for r in runs])
        mean_h    = histories.mean(axis=0)
        std_h     = histories.std(axis=0)
        iters     = np.arange(len(mean_h))

        color = COLORS[vname]
        label = LABELS[vname]
        ax.plot(iters, mean_h, label=label, color=color, linewidth=2)
        ax.fill_between(iters, mean_h - std_h, mean_h + std_h,
                        alpha=0.15, color=color)

    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("Best Score (Total Waiting Time)", fontsize=13)
    ax.set_title("PSO Variant Convergence Comparison\n"
                 f"(mean ± 1 std over {N_RUNS} runs)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, "variant_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] convergence to {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Plot: boxplots for pairwise comparisons
# ─────────────────────────────────────────────────────────────────────────────

def _boxplot(all_results, variant_names, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    data   = [
        [r["score"] for r in all_results[v]]
        for v in variant_names
    ]
    labels = [LABELS[v] for v in variant_names]
    colors = [COLORS[v] for v in variant_names]

    bp = ax.boxplot(data, patch_artist=True, notch=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Final Best Score", fontsize=12)
    ax.set_title(title + f"\n({N_RUNS} runs per variant)", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] boxplot to {path}")
    return path


def plot_inertia_boxplot(all_results):
    return _boxplot(
        all_results,
        ["constant_inertia", "linear_inertia"],
        "Inertia Strategy Comparison",
        "inertia_boxplot.png",
    )


def plot_topology_boxplot(all_results):
    return _boxplot(
        all_results,
        ["global_topology", "local_topology"],
        "Topology Comparison",
        "topology_boxplot.png",
    )


def plot_all_boxplot(all_results):
    return _boxplot(
        all_results,
        EXPERIMENT_VARIANTS,
        "All Variants Final Score Distribution",
        "all_variants_boxplot.png",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Print console summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(all_results):
    header = f"{'Variant':<30} {'Mean':>10} {'Std':>8} {'Best':>10} {'Worst':>10}"
    print(f"\n{'═'*70}")
    print("  VARIANT COMPARISON SUMMARY")
    print(f"{'═'*70}")
    print(header)
    print("─" * 70)
    for vname, runs in all_results.items():
        scores = [r["score"] for r in runs]
        print(f"  {LABELS[vname]:<28} "
              f"{np.mean(scores):>10.1f} "
              f"{np.std(scores):>8.1f} "
              f"{np.min(scores):>10.1f} "
              f"{np.max(scores):>10.1f}")
    print(f"{'═'*70}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  PSO Variant Experiments — Person 4")
    print(f"  Variants: {EXPERIMENT_VARIANTS}")
    print(f"  Runs per variant: {N_RUNS}")
    print(f"  Seeds: stored in results/seeds.csv")
    print("=" * 55)

    save_seeds()
    all_results = run_all_variants()

    print_summary_table(all_results)
    save_summary(all_results)
    plot_convergence(all_results)
    plot_inertia_boxplot(all_results)
    plot_topology_boxplot(all_results)
    plot_all_boxplot(all_results)

    print("\n✅  All done. Results written to ./results/")