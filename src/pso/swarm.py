import numpy as np
from .particle import Particle


class Swarm:
    """
    PSO Swarm — manages all particles and tracks the global best.

    Supports two inertia strategies:
        - 'constant'  : w stays fixed throughout the run
        - 'linear'    : w decreases linearly from w_max → w_min

    Supports two topology strategies:
        - 'global' : every particle is attracted to the single best in the whole swarm
        - 'local'  : every particle is attracted to the best in its neighbourhood (ring)
    """

    def __init__(
        self,
        num_particles: int,
        num_intersections: int,
        min_green: float = 5.0,
        max_green: float = 60.0,
        inertia_strategy: str = "constant",   # 'constant' | 'linear'
        topology: str = "global",             # 'global'   | 'local'
        w: float = 0.5,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 1.5,
        c2: float = 1.5,
        max_iterations: int = 100,
        neighborhood_size: int = 3,
    ):
        self.num_particles = num_particles
        self.num_intersections = num_intersections
        self.min_green = min_green
        self.max_green = max_green

        # ── inertia ──────────────────────────────────────────────
        self.inertia_strategy = inertia_strategy
        self.w = w
        self.w_max = w_max
        self.w_min = w_min

        # ── topology ─────────────────────────────────────────────
        self.topology = topology
        self.neighborhood_size = neighborhood_size   # used only for 'local'

        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.current_iteration = 0

        # ── global best ──────────────────────────────────────────
        self.gbest_position = None
        self.gbest_score = float("inf")

        # ── particles ────────────────────────────────────────────
        self.particles: list[Particle] = [
            Particle(num_intersections, min_green, max_green)
            for _ in range(num_particles)
        ]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _current_inertia(self) -> float:
        """Return the inertia weight for the current iteration."""
        if self.inertia_strategy == "linear":
            # linearly decreases from w_max to w_min
            ratio = self.current_iteration / max(self.max_iterations - 1, 1)
            return self.w_max - ratio * (self.w_max - self.w_min)
        return self.w  # constant

    def _local_best_position(self, idx: int) -> np.ndarray:
        """
        Return the best position found by the ring-neighbourhood of particle idx.
        Neighbourhood includes idx itself plus (neighborhood_size//2) on each side.
        """
        half = self.neighborhood_size // 2
        n = self.num_particles
        best_score = float("inf")
        best_pos = self.particles[idx].pbest_position

        for offset in range(-half, half + 1):
            neighbour = self.particles[(idx + offset) % n]
            if neighbour.pbest_score < best_score:
                best_score = neighbour.pbest_score
                best_pos = neighbour.pbest_position

        return best_pos

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def update_gbest(self, scores: list[float]) -> None:
        """Update global best using freshly evaluated scores."""
        for particle, score in zip(self.particles, scores):
            # update particle's personal best
            if score < particle.pbest_score:
                particle.pbest_score = score
                particle.pbest_position = particle.position.copy()

            # update swarm global best
            if score < self.gbest_score:
                self.gbest_score = score
                self.gbest_position = particle.position.copy()

    def step(self, scores: list[float]) -> None:
        """
        One PSO iteration:
          1. Update personal & global bests from this iteration's scores.
          2. Update each particle's velocity & position.
        """
        self.update_gbest(scores)

        w = self._current_inertia()

        for idx, particle in enumerate(self.particles):
            # choose attractor based on topology
            if self.topology == "local":
                guide = self._local_best_position(idx)
            else:  # global
                guide = self.gbest_position

            particle.update_velocity(
                guide,
                inertia=w,
                c1=self.c1,
                c2=self.c2,
            )
            particle.update_position(self.min_green, self.max_green)

        self.current_iteration += 1

    def get_positions(self) -> list[np.ndarray]:
        """Return current position (candidate solution) of every particle."""
        return [p.position for p in self.particles]