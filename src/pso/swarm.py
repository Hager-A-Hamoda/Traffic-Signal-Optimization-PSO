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

    Diversity Preservation — Crowding:
        - 'none'     : standard PSO, no crowding
        - 'crowding' : a new particle replaces the most similar (closest) particle
                       in the swarm instead of the worst, keeping solutions spread out
    """

    def __init__(
        self,
        num_particles: int,
        num_intersections: int,
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
        max_iterations: int = 100,
        neighborhood_size: int = 3,
    ):
        self.num_particles = num_particles
        self.num_intersections = num_intersections
        self.min_green = min_green
        self.max_green = max_green

        self.inertia_strategy = inertia_strategy
        self.w = w
        self.w_max = w_max
        self.w_min = w_min

        self.topology = topology
        self.neighborhood_size = neighborhood_size

        self.diversity = diversity

        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.current_iteration = 0

        self.gbest_position = None
        self.gbest_score = float("inf")

        self.particles: list[Particle] = [
            Particle(num_intersections, min_green, max_green)
            for _ in range(num_particles)
        ]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _current_inertia(self) -> float:
        if self.inertia_strategy == "linear":
            ratio = self.current_iteration / max(self.max_iterations - 1, 1)
            return self.w_max - ratio * (self.w_max - self.w_min)
        return self.w

    def _local_best_position(self, idx: int) -> np.ndarray:
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

    def _crowding_replace(self, candidate_position: np.ndarray, candidate_score: float) -> None:
        """
        Crowding — Diversity Preservation:

        Instead of the candidate particle simply updating itself,
        it finds the most SIMILAR particle in the swarm (smallest
        Euclidean distance) and replaces it only if the candidate
        is better. This stops particles from clustering in one spot.

        Steps:
          1. Measure distance from candidate to every particle.
          2. Find the closest particle (most similar solution).
          3. Replace it only if candidate_score is better.
        """
        distances = [
            np.linalg.norm(candidate_position - p.position)
            for p in self.particles
        ]
        closest_idx = int(np.argmin(distances))
        closest = self.particles[closest_idx]

        if candidate_score < closest.pbest_score:
            closest.position = candidate_position.copy()
            closest.velocity = np.zeros(self.num_intersections)
            closest.pbest_position = candidate_position.copy()
            closest.pbest_score = candidate_score

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def update_gbest(self, scores: list[float]) -> None:
        for particle, score in zip(self.particles, scores):
            if score < particle.pbest_score:
                particle.pbest_score = score
                particle.pbest_position = particle.position.copy()

            if score < self.gbest_score:
                self.gbest_score = score
                self.gbest_position = particle.position.copy()

    def step(self, scores: list[float]) -> None:
        """
        One PSO iteration:
          1. Update personal & global bests.
          2. Move each particle (velocity + position update).
          3. If crowding enabled, apply crowding replacement.
        """
        self.update_gbest(scores)

        w = self._current_inertia()

        for idx, particle in enumerate(self.particles):
            if self.topology == "local":
                guide = self._local_best_position(idx)
            else:
                guide = self.gbest_position

            particle.update_velocity(guide, inertia=w, c1=self.c1, c2=self.c2)
            particle.update_position(self.min_green, self.max_green)

            # ── crowding replacement ──────────────────────────────
            if self.diversity == "crowding":
                self._crowding_replace(particle.position, scores[idx])

        self.current_iteration += 1

    def get_positions(self) -> list[np.ndarray]:
        return [p.position for p in self.particles]