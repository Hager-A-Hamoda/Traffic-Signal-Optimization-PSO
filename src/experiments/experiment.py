import time
import numpy as np

from src.pso.fitness import FitnessEvaluator
from src.pso.variants import build_variant



class Experiment:
    def __init__(
        self,
        name: str,
        variant: str,
        params: dict | None = None,
        num_runs: int = 30,
        steps: int = 100,
        seeds: list[int] | None = None,
    ):
        self.name = name
        self.variant = variant
        self.params = params or {}
        self.num_runs = num_runs
        self.steps = steps

        # Generate or accept seeds
        if seeds is not None:
            assert len(seeds) >= num_runs, "Not enough seeds supplied."
            self.seeds = seeds[:num_runs]
        else:
            rng = np.random.default_rng(seed=0)
            self.seeds = rng.integers(0, 2**31, size=num_runs).tolist()  # random seeds for reproducibility 

        # Results container (populated after run())
        self.results: list[dict] = []
        self._summary: dict | None = None

    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> dict:
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Experiment : {self.name}")
            print(f"  Variant    : {self.variant}")
            print(f"  Runs       : {self.num_runs}")
            print(f"{'='*60}")

        evaluator = FitnessEvaluator(steps=self.steps)
        self.results = []

        for run_idx, seed in enumerate(self.seeds):
            t0 = time.time()

            opt = build_variant(
                self.variant,
                fitness_evaluator=evaluator,
                seed=int(seed),
                **self.params,
            )
            result = opt.run()
            runtime = time.time() - t0

            num_intersections = len(result["best_position"])
            num_stops = num_intersections
            avg_queue_length = float(result["best_score"] / max(num_intersections, 1))
            avg_waiting_time = avg_queue_length / max(num_intersections, 1)
            
            run_record = {
                "run": run_idx + 1,
                "seed": int(seed),
                "best_score": float(result["best_score"]),
                "best_position": result["best_position"].tolist(),
                "history": [float(v) for v in result["history"]],
                "runtime_s": round(runtime, 3),
                "avg_queue_length": round(avg_queue_length, 3),
                "num_stops": num_stops,
                "avg_waiting_time": avg_waiting_time,
            }
            self.results.append(run_record)

            if verbose:
                print(
                    f"  Run {run_idx+1:2d}/{self.num_runs}  "
                    f"score={result['best_score']:.0f}  "
                    f"timings={result['best_position'].round(1).tolist()}  "
                    f"({runtime:.2f}s)"
                )

        self._summary = self._compute_summary()
        if verbose:
            self._print_summary()
        return self._summary

    # ------------------------------------------------------------------

    def _compute_summary(self) -> dict:
        scores = [r["best_score"] for r in self.results]
        runtimes = [r["runtime_s"] for r in self.results]

        best_run = min(self.results, key=lambda r: r["best_score"])

        queue_lengths = [r["avg_queue_length"] for r in self.results]

        waiting_times = [r["avg_waiting_time"] for r in self.results]
        num_stops_list = [r["num_stops"] for r in self.results]

        return {
            "experiment_name": self.name,
            "variant": self.variant,
            "num_runs": self.num_runs,
            "seeds": self.seeds,
            "params": self.params,
            # fitness statistics
            "best_score": float(best_run["best_score"]),
            "best_position": best_run["best_position"],
            "best_run_index": best_run["run"],
            "avg_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "median_score": float(np.median(scores)),
            "worst_score": float(np.max(scores)),
            # runtime
            "avg_runtime_s": float(np.mean(runtimes)),
            "total_runtime_s": float(np.sum(runtimes)),
            # queue length metric (existing)
            "avg_queue_length": round(float(np.mean(queue_lengths)), 3),
            "std_queue_length": round(float(np.std(queue_lengths)), 3),
            # NEW: average waiting time
            "avg_waiting_time": round(float(np.mean(waiting_times)), 3),
            "std_waiting_time": round(float(np.std(waiting_times)), 3),
            # NEW: number of stops (intersections)
            "num_stops": int(np.mean(num_stops_list)),
            # raw per-run records
            "runs": self.results,
        }

    def _print_summary(self):
        s = self._summary
        print(f"\n  -- Summary ----------------------------------")
        print(f"  Best  : {s['best_score']:.0f}  @ run {s['best_run_index']}")
        print(f"  Avg   : {s['avg_score']:.1f}  +/- {s['std_score']:.1f}")
        print(f"  Median: {s['median_score']:.0f}")
        print(f"  Worst : {s['worst_score']:.0f}")
        print(f"  Time  : {s['avg_runtime_s']:.2f}s / run")
        print(f"  Queue : {s['avg_queue_length']:.3f} +/- {s['std_queue_length']:.3f} (avg queue length)")
        print(f"  Wait  : {s['avg_waiting_time']:.3f} +/- {s['std_waiting_time']:.3f} (avg waiting time)")
        print(f"  Stops : {s['num_stops']} intersections")

    @property  # allows access via exp.summary instead of exp.summary() to avoid confusion with the summary dict and error if called before run()
    def summary(self) -> dict:  
        if self._summary is None:
            raise RuntimeError("Call run() first.")
        return self._summary
