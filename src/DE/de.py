import numpy as np

class DEOptimizer:
    def __init__(
        self,
        fitness_evaluator,
        num_intersections: int = 3,
        population_size: int = 30,
        max_iterations: int = 100,
        min_green: float = 5.0,
        max_green: float = 60.0,
        F: float = 0.8,
        CR: float = 0.9,
        representation: str = "continuous",   # 'continuous' or 'discrete'
        init_strategy: str = "random",        # 'random' or 'uniform'
        mutation_strategy: str = "rand1",     # 'rand1' or 'best1'
        crossover_strategy: str = "binomial", # 'binomial' or 'exponential'
        parent_selection: str = "random",     # 'random' or 'tournament'
        seed: int = None,
    ):
        self.fitness_evaluator = fitness_evaluator
        self.num_intersections = num_intersections
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.min_green = min_green
        self.max_green = max_green
        self.F = F
        self.CR = CR
        self.representation = representation
        self.init_strategy = init_strategy
        self.mutation_strategy = mutation_strategy
        self.crossover_strategy = crossover_strategy
        self.parent_selection = parent_selection
        self.history = []

        if seed is not None:
            np.random.seed(seed)

        self.population = self._initialise()

    # Initialisation Strategy 1 — Random: generates solutions randomly across the search space
    # Initialisation Strategy 2 — Uniform: distributes solutions evenly across the search space
    def _initialise(self):
        if self.init_strategy == "uniform":
            population = []
            for i in range(self.population_size):
                individual = self.min_green + (self.max_green - self.min_green) * i / self.population_size
                solution = np.full(self.num_intersections, individual)
                population.append(solution)
            return np.array(population, dtype=float)
        else:
            return np.random.uniform(
                self.min_green, self.max_green,
                (self.population_size, self.num_intersections)
            )

    # Representation 1 — Continuous: real-valued timings (e.g. 30.5, 42.7)
    # Representation 2 — Discrete: integer-valued timings only (e.g. 30, 42)
    def _represent(self, individual):
        if self.representation == "discrete":
            return np.round(individual).astype(float)
        return individual

    # Parent Selection 1 — Random: selects 3 individuals randomly from the population
    # Parent Selection 2 — Tournament: selects the best individual from each random group of 3
    def _select_parents(self, idx):
        candidates = [i for i in range(self.population_size) if i != idx]

        if self.parent_selection == "tournament":
            selected = []
            scores = [self.fitness_evaluator.evaluate(
                self._represent(self.population[i])) for i in range(self.population_size)]
            for _ in range(3):
                group = np.random.choice(candidates, 3, replace=False)
                group_scores = [scores[i] for i in group]
                selected.append(group[np.argmin(group_scores)])
            return selected
        else:
            return np.random.choice(candidates, 3, replace=False)

    # Mutation Strategy 1 — rand/1: adds weighted difference of two random individuals to a third random individual
    # Mutation Strategy 2 — best/1: adds weighted difference of two random individuals to the current global best
    def _mutate(self, idx, scores):
        a, b, c = self._select_parents(idx)

        if self.mutation_strategy == "best1":
            best_idx = np.argmin(scores)
            mutant = self.population[best_idx] + self.F * (self.population[b] - self.population[c])
        else:
            mutant = self.population[a] + self.F * (self.population[b] - self.population[c])

        mutant = np.clip(mutant, self.min_green, self.max_green)
        return mutant

    # Crossover Strategy 1 — Binomial: each parameter is independently crossed over with probability CR
    # Crossover Strategy 2 — Exponential: crosses over a contiguous block of parameters starting at a random point
    def _crossover(self, target, mutant):
        trial = np.copy(target)

        if self.crossover_strategy == "exponential":
            start = np.random.randint(self.num_intersections)
            j = start
            while True:
                trial[j] = mutant[j]
                j = (j + 1) % self.num_intersections
                if np.random.random() >= self.CR or j == start:
                    break
        else:
            for j in range(self.num_intersections):
                if np.random.random() < self.CR:
                    trial[j] = mutant[j]

        return trial

    # Selection: replaces the target individual only if the trial solution achieves a better or equal fitness score
    def _select(self, target, trial, target_score, trial_score):
        if trial_score <= target_score:
            return trial, trial_score
        return target, target_score

    def run(self):
        scores = [self.fitness_evaluator.evaluate(
            self._represent(ind)) for ind in self.population]

        best_idx = np.argmin(scores)
        best_position = self._represent(self.population[best_idx].copy())
        best_score = scores[best_idx]
        self.history.append(best_score)

        for _ in range(self.max_iterations):
            new_population = []
            new_scores = []

            for idx in range(self.population_size):
                target = self.population[idx]
                target_score = scores[idx]

                mutant = self._mutate(idx, scores)
                trial = self._crossover(target, mutant)
                trial = self._represent(trial)
                trial_score = self.fitness_evaluator.evaluate(trial)

                winner, winner_score = self._select(
                    target, trial, target_score, trial_score)

                new_population.append(winner)
                new_scores.append(winner_score)

                if winner_score < best_score:
                    best_score = winner_score
                    best_position = self._represent(winner.copy())

            self.population = np.array(new_population)
            scores = new_scores
            self.history.append(best_score)

        return {
            "best_position": best_position,
            "best_score": best_score,
            "history": self.history,
        }