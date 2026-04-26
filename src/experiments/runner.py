import time
import json
import csv
import os
import numpy as np

from .experiment import Experiment


# ---------------------------------------------------------------------------
# Standard experiment configurations
# ---------------------------------------------------------------------------
STANDARD_CONFIGS = [
    {
        "name": "Baseline (Constant Inertia, Global)",
        "variant": "constant_inertia",
        "params": {"num_particles": 20, "max_iterations": 50},
    },
    {
        "name": "Linear Inertia, Global",
        "variant": "linear_inertia",
        "params": {"num_particles": 20, "max_iterations": 50},
    },
    {
        "name": "Constant Inertia, Local Topology",
        "variant": "local_topology",
        "params": {"num_particles": 20, "max_iterations": 50},
    },
    {
        "name": "Linear Inertia, Local Topology",
        "variant": "linear_local",
        "params": {"num_particles": 20, "max_iterations": 50},
    },
    {
        "name": "Linear Inertia, Global (High c)",
        "variant": "linear_global",
        "params": {"num_particles": 20, "max_iterations": 50},
    },
    {
        "name": "Constriction Factor PSO",
        "variant": "constriction",
        "params": {"num_particles": 20, "max_iterations": 50},
    },
]


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------
class ExperimentRunner:
    def __init__(
        self,
        configs: list[dict] | None = None,
        num_runs: int = 30,
        steps: int = 100,
    ):
        self.configs = configs if configs is not None else STANDARD_CONFIGS
        self.num_runs = num_runs
        self.steps = steps
        self.experiments: list[Experiment] = []
        self.all_results: list[dict] = []

    # ------------------------------------------------------------------

    def run_all(self, verbose: bool = True) -> list[dict]:
        self.experiments = []
        self.all_results = []

        total_start = time.time()

        for cfg in self.configs:
            exp = Experiment(
                name=cfg["name"],
                variant=cfg["variant"],
                params=cfg.get("params", {}),
                num_runs=self.num_runs,
                steps=self.steps,
            )
            summary = exp.run(verbose=verbose)
            self.experiments.append(exp)
            self.all_results.append(summary)

        total_time = time.time() - total_start

        if verbose:
            print(f"\n{'='*60}")
            print(f"  All experiments complete in {total_time/60:.1f} min")
            self.print_comparison()

        return self.all_results

    # ------------------------------------------------------------------

    def print_comparison(self):
        if not self.all_results:
            print("No results yet. Call run_all() first.")
            return

        sorted_results = sorted(self.all_results, key=lambda r: r["avg_score"])
        baseline_avg = self.all_results[0]["avg_score"]

        print(f"\n{'='*60}")
        print(f"  COMPARISON TABLE  (sorted by avg score, lower = better)")
        print(f"{'='*60}")
        header = f"  {'Experiment':<38} {'Best':>7} {'Avg':>8} {'Std':>7} {'Time':>6}"
        print(header)
        print(f"  {'-'*60}")

        for r in sorted_results:
            improvement = (baseline_avg - r["avg_score"]) / baseline_avg * 100
            flag = f"(+{improvement:.1f}%)" if improvement > 0 else ""
            print(
                f"  {r['experiment_name']:<38} "
                f"{r['best_score']:>7.0f} "
                f"{r['avg_score']:>8.1f} "
                f"{r['std_score']:>7.1f} "
                f"{r['avg_runtime_s']:>5.2f}s  {flag}"
            )

        best = sorted_results[0]
        print(f"\n  Best config: {best['experiment_name']}")
        print(f"  Avg score  : {best['avg_score']:.1f} +/- {best['std_score']:.1f}")
        print(f"  Best timing: {[round(v,1) for v in best['best_position']]}")

    # ------------------------------------------------------------------

    def save_results(self, path: str = "results/experiments.json"):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.all_results, f, indent=2)
        print(f"\n  Results saved -> {path}")

    def save_csv(self, path: str = "results/experiments.csv"):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        fields = [
            "experiment_name", "variant", "num_runs",
            "best_score", "avg_score", "std_score",
            "median_score", "worst_score",
            "avg_runtime_s", "total_runtime_s",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in self.all_results:
                writer.writerow({k: r[k] for k in fields})
        print(f"  CSV saved    -> {path}")

    def save_seeds(self, path: str = "results/seeds.json"):
        seed_data = {r["experiment_name"]: r["seeds"] for r in self.all_results}
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(seed_data, f, indent=2)
        print(f"  Seeds saved  -> {path}")
