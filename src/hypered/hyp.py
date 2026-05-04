import numpy as np
from src.pso.particle import Particle
from src.pso.fitness import FitnessEvaluator

class HybridPSODE:
    """
    Hybrid PSO-DE Optimizer:
    
    Combines the memory mechanism of PSO (pbest, gbest) 
    with the mutation and crossover operators of DE to 
    generate new candidate solutions.
    
    Each particle:
    - Uses DE mutation to explore new regions
    - Uses DE crossover to mix solutions
    - Keeps PSO memory (pbest, gbest) to guide the search
    """

    def __init__(
        self,
        fitness_evaluator,
        num_intersections: int = 3,
        num_particles: int = 30,
        max_iterations: int = 100,
        min_green: float = 5.0,
        max_green: float = 60.0,
        F: float = 0.8,        # DE mutation factor
        CR: float = 0.9,       # DE crossover rate
        w: float = 0.5,        # PSO inertia weight (used in gbest guidance)
        seed: int = None,
    ):
        self.fitness_evaluator = fitness_evaluator
        self.num_intersections = num_intersections
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.min_green = min_green
        self.max_green = max_green
        self.F = F
        self.CR = CR
        self.w = w
        self.history = []

        if seed is not None:
            np.random.seed(seed)

        # Initialise particles (PSO memory)
        self.particles = [
            Particle(num_intersections, min_green, max_green)
            for _ in range(num_particles)
        ]

        # Global best (from PSO)
        self.gbest_position = None
        self.gbest_score = float("inf")

    # ------------------------------------------------------------------

    def _de_mutation(self, idx, positions):
        """
        DE Mutation — rand/1 strategy:
        Uses three random particles to create a mutant vector.
        Replaces the standard PSO velocity update.
        """
        candidates = [i for i in range(self.num_particles) if i != idx]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = positions[a] + self.F * (positions[b] - positions[c])
        mutant = np.clip(mutant, self.min_green, self.max_green)
        return mutant

    def _de_crossover(self, target, mutant):
        """
        DE Crossover — binomial strategy:
        Mixes the mutant vector with the target using crossover rate CR.
        """
        trial = np.copy(target)
        for j in range(self.num_intersections):
            if np.random.random() < self.CR:
                trial[j] = mutant[j]
        return trial

    def _pso_guidance(self, particle, trial):
        """
        PSO Guidance:
        Pulls the trial solution towards pbest and gbest.
        """
        r1 = np.random.random(self.num_intersections)
        r2 = np.random.random(self.num_intersections)

        guided = trial + (
            r1 * (particle.pbest_position - trial) +
            r2 * (self.gbest_position - trial)
        ) * self.w

        guided = np.clip(guided, self.min_green, self.max_green)
        return guided

    def _update_bests(self, particle, score):
        """
        Updates personal best (PSO memory) and global best.
        """
        if score < particle.pbest_score:
            particle.pbest_score = score
            particle.pbest_position = particle.position.copy()

        if score < self.gbest_score:
            self.gbest_score = score
            self.gbest_position = particle.position.copy()

    # ------------------------------------------------------------------

    def run(self):
        # Evaluate initial population
        positions = [p.position for p in self.particles]
        scores = [self.fitness_evaluator.evaluate(pos) for pos in positions]

        # Initialise bests
        for particle, score in zip(self.particles, scores):
            self._update_bests(particle, score)

        self.history.append(self.gbest_score)

        # Main loop
        for _ in range(self.max_iterations):
            positions = [p.position for p in self.particles]

            for idx, particle in enumerate(self.particles):

                # Step 1: DE Mutation — explores new regions
                mutant = self._de_mutation(idx, positions)

                # Step 2: DE Crossover — mixes mutant with current position
                trial = self._de_crossover(particle.position, mutant)

                # Step 3: PSO Guidance — pulls trial towards pbest and gbest
                guided = self._pso_guidance(particle, trial)

                # Step 4: Evaluate new candidate
                guided_score = self.fitness_evaluator.evaluate(guided)

                # Step 5: Selection — keep better solution (DE selection)
                if guided_score < scores[idx]:
                    particle.position = guided.copy()
                    scores[idx] = guided_score
                    self._update_bests(particle, guided_score)

            self.history.append(self.gbest_score)

        return {
            "best_position": self.gbest_position,
            "best_score": self.gbest_score,
            "history": self.history,
        }