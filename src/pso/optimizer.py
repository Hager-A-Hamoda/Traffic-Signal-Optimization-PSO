import numpy as np
from .swarm import Swarm


class PSOOptimizer:
    """
    Runs PSO on the traffic-signal timing problem.

    Parameters
    ----------
    fitness_evaluator : FitnessEvaluator
        Object with an ``evaluate(position) -> float`` method.
    num_intersections : int
        Dimensionality of the search space (= number of traffic lights).
    num_particles : int
        Swarm size.
    max_iterations : int
        Stopping criterion (number of PSO iterations).
    min_green / max_green : float
        Search-space bounds for each timing variable.
    inertia_strategy : str
        'constant' or 'linear'  (passed to Swarm).
    topology : str
        'global' or 'local'  (passed to Swarm).
    diversity : str
        'none' or 'crowding'  (passed to Swarm).
    w, w_max, w_min : float
        Inertia-weight parameters (passed to Swarm).
    c1, c2 : float
        Cognitive / social acceleration coefficients.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        fitness_evaluator,
        num_intersections: int = 3,
        num_particles: int = 30,
        max_iterations: int = 100,
        min_green: float = 5.0,
        max_green: float = 60.0,
        inertia_strategy: str = "constant",
        topology: str = "global",
        diversity: str = "none",              # 'none' | 'crowding'
        w: float = 0.5,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 1.5,
        c2: float = 1.5,
        seed: int | None = None,
    ):
        self.fitness_evaluator = fitness_evaluator
        self.num_intersections = num_intersections
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.min_green = min_green
        self.max_green = max_green
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        self.swarm = Swarm(
            num_particles=num_particles,
            num_intersections=num_intersections,
            min_green=min_green,
            max_green=max_green,
            inertia_strategy=inertia_strategy,
            topology=topology,
            diversity=diversity,
            w=w,
            w_max=w_max,
            w_min=w_min,
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
        )

        # ── history (for analysis / plotting) ────────────────────
        self.history: list[float] = []   # gbest score per iteration

    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute the full PSO run.

        Returns
        -------
        dict with keys:
            best_position  – np.ndarray of optimal green-time values
            best_score     – corresponding fitness (total waiting time)
            history        – list of best scores per iteration
        """
        # ── evaluate initial positions ────────────────────────────
        positions = self.swarm.get_positions()
        scores = [self.fitness_evaluator.evaluate(pos) for pos in positions]

        # initialise personal bests before the first step
        self.swarm.update_gbest(scores)
        self.history.append(self.swarm.gbest_score)

        # ── main loop ─────────────────────────────────────────────
        for _ in range(self.max_iterations):
            # 1. move particles
            self.swarm.step(scores)

            # 2. evaluate new positions
            positions = self.swarm.get_positions()
            scores = [self.fitness_evaluator.evaluate(pos) for pos in positions]

            # 3. record best
            self.history.append(self.swarm.gbest_score)

        return {
            "best_position": self.swarm.gbest_position,
            "best_score": self.swarm.gbest_score,
            "history": self.history,
        }