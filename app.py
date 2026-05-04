"""
Traffic Signal Optimization — Flask GUI
Run: python app.py
"""

import json
import threading
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
import queue
import time

app = Flask(__name__)

# Global state
progress_queues = {}
results_store = {}


def run_optimization(job_id, algo, params):
    """Run optimization in background thread, stream progress via queue."""
    q = progress_queues[job_id]

    try:
        from src.pso.fitness import FitnessEvaluator
        evaluator = FitnessEvaluator(steps=params.get("steps", 100))

        num_intersections = 3
        history = []
        best_position = None
        best_score = float("inf")

        if algo == "pso":
            from src.pso.optimizer import PSOOptimizer
            opt = PSOOptimizer(
                fitness_evaluator=evaluator,
                num_intersections=num_intersections,
                num_particles=params.get("num_particles", 20),
                max_iterations=params.get("max_iterations", 50),
                inertia_strategy=params.get("inertia_strategy", "linear"),
                topology=params.get("topology", "global"),
                w=params.get("w", 0.5),
                w_max=params.get("w_max", 0.9),
                w_min=params.get("w_min", 0.4),
                c1=params.get("c1", 1.5),
                c2=params.get("c2", 1.5),
                seed=params.get("seed", 42),
            )

        elif algo == "de":
            from src.DE.de import DEOptimizer
            opt = DEOptimizer(
                fitness_evaluator=evaluator,
                num_intersections=num_intersections,
                population_size=params.get("num_particles", 20),
                max_iterations=params.get("max_iterations", 50),
                F=params.get("F", 0.8),
                CR=params.get("CR", 0.9),
                mutation_strategy=params.get("mutation_strategy", "rand1"),
                crossover_strategy=params.get("crossover_strategy", "binomial"),
                representation=params.get("representation", "continuous"),
                init_strategy=params.get("init_strategy", "random"),
                parent_selection=params.get("parent_selection", "random"),
                seed=params.get("seed", 42),
            )

        elif algo == "hybrid":
            from src.hypered.hyp import HybridPSODE
            opt = HybridPSODE(
                fitness_evaluator=evaluator,
                num_intersections=num_intersections,
                num_particles=params.get("num_particles", 20),
                max_iterations=params.get("max_iterations", 50),
                F=params.get("F", 0.6),
                CR=params.get("CR", 0.8),
                w=params.get("w", 0.4),
                seed=params.get("seed", 42),
            )

        # Run optimizer then stream history
        result = opt.run()
        for i, score in enumerate(result["history"]):
            q.put({"iter": i, "score": float(score), "total": len(result["history"])})

        # Baseline score (fixed timings [20,20,20])
        from src.simulation.network_builder import build_network
        net = build_network()
        net.reset()
        baseline = net.simulate([20, 20, 20])

        improvement = (baseline - result["best_score"]) / baseline * 100

        results_store[job_id] = {
            "algo": algo,
            "best_score": float(result["best_score"]),
            "best_position": [round(float(v), 1) for v in result["best_position"]],
            "baseline_score": float(baseline),
            "improvement": round(improvement, 2),
            "history": [float(v) for v in result["history"]],
        }

        q.put({"done": True, "job_id": job_id})

    except Exception as e:
        q.put({"error": str(e)})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    data = request.json
    algo = data.get("algo", "pso")
    params = data.get("params", {})

    job_id = f"{algo}_{int(time.time() * 1000)}"
    progress_queues[job_id] = queue.Queue()

    thread = threading.Thread(target=run_optimization, args=(job_id, algo, params))
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream(job_id):
    def generate():
        if job_id not in progress_queues:
            yield f"data: {json.dumps({'error': 'job not found'})}\n\n"
            return
        q = progress_queues[job_id]
        while True:
            try:
                msg = q.get(timeout=30)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("done") or msg.get("error"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'ping': True})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/results/<job_id>")
def get_results(job_id):
    if job_id in results_store:
        return jsonify(results_store[job_id])
    return jsonify({"error": "not ready"}), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
