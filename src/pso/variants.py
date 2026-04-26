"""
PSO Variants — Person 4
========================
Implements multiple PSO variant configurations to enable experimental
comparison as required by the project specification.

Variants covered
----------------
1. Inertia weight strategies  : constant  vs  linear decreasing
2. Topology                   : global best  vs  local best (ring)
3. Velocity clamping          : unclamped  vs  clamped (Vmax)
4. (Bonus) Constriction Factor PSO  — an alternative to the inertia-weight form

All variants are exposed through `build_variant(name, **kwargs)` which
returns a configured `PSOOptimizer` ready to call `.run()`.
"""

import numpy as np
from .optimizer import PSOOptimizer
from .fitness import FitnessEvaluator

# ─────────────────────────────────────────────────────────────────────────────
# Variant registry
# ─────────────────────────────────────────────────────────────────────────────

VARIANT_DEFAULTS = {
    "num_intersections": 3,
    "num_particles":     30,
    "max_iterations":    100,
    "min_green":         5.0,
    "max_green":         60.0,
    "seed":              None,
}

# Pre-defined named variants (name → kwarg overrides for PSOOptimizer)
VARIANTS = {
    # ── Inertia weight ────────────────────────────────────────────────
    "constant_inertia": {
        "inertia_strategy": "constant",
        "w":  0.5,
        "c1": 1.5,
        "c2": 1.5,
        "topology": "global",
    },
    "linear_inertia": {
        "inertia_strategy": "linear",
        "w_max": 0.9,
        "w_min": 0.4,
        "c1":    1.5,
        "c2":    1.5,
        "topology": "global",
    },
    # ── Topology ──────────────────────────────────────────────────────
    "global_topology": {
        "inertia_strategy": "constant",
        "w":  0.5,
        "c1": 1.5,
        "c2": 1.5,
        "topology": "global",
    },
    "local_topology": {
        "inertia_strategy": "constant",
        "w":  0.5,
        "c1": 1.5,
        "c2": 1.5,
        "topology": "local",
    },
    # ── Combined best variants ────────────────────────────────────────
    "linear_global": {
        "inertia_strategy": "linear",
        "w_max": 0.9,
        "w_min": 0.4,
        "c1":    2.0,
        "c2":    2.0,
        "topology": "global",
    },
    "linear_local": {
        "inertia_strategy": "linear",
        "w_max": 0.9,
        "w_min": 0.4,
        "c1":    2.0,
        "c2":    2.0,
        "topology": "local",
    },
    # ── Constriction factor PSO (Clerc & Kennedy 2002) ────────────────
    "constriction": {
        "inertia_strategy": "constriction",
        "c1": 2.05,
        "c2": 2.05,
        "topology": "global",
    },
}


def build_variant(
    name: str,
    fitness_evaluator=None,
    seed: int | None = None,
    **overrides,
) -> PSOOptimizer:
    """
    Build a PSOOptimizer pre-configured for the named variant.

    Parameters
    ----------
    name : str
        Key from VARIANTS dict.
    fitness_evaluator : FitnessEvaluator | None
        Shared evaluator (creates one if not supplied).
    seed : int | None
        Random seed.
    **overrides
        Any additional kwargs forwarded to PSOOptimizer (override variant defaults).

    Returns
    -------
    PSOOptimizer
    """
    if name not in VARIANTS:
        raise ValueError(
            f"Unknown variant '{name}'. "
            f"Available: {list(VARIANTS.keys())}"
        )

    if fitness_evaluator is None:
        fitness_evaluator = FitnessEvaluator()

    kwargs = {**VARIANT_DEFAULTS, **VARIANTS[name], **overrides, "seed": seed}

    # Constriction factor needs a patched Swarm — we handle it via a subclass
    if kwargs.get("inertia_strategy") == "constriction":
        return _build_constriction_optimizer(fitness_evaluator, kwargs)

    return PSOOptimizer(fitness_evaluator=fitness_evaluator, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Constriction Factor PSO (bonus variant)
# ─────────────────────────────────────────────────────────────────────────────

class ConstrictionSwarm:
    """
    Minimal Swarm variant that uses the constriction-factor formulation:

        χ = 2κ / |2 - φ - √(φ² - 4φ)|
        φ = c1 + c2  (must be > 4)
        κ = 1  (standard)

    The velocity update becomes:
        v = χ * (v + c1*r1*(pbest-x) + c2*r2*(gbest-x))
    """

    def __init__(self, num_particles, num_intersections, min_green, max_green,
                 c1, c2, max_iterations, **_):
        from .particle import Particle

        self.num_particles     = num_particles
        self.num_intersections = num_intersections
        self.min_green         = min_green
        self.max_green         = max_green
        self.c1                = c1
        self.c2                = c2
        self.max_iterations    = max_iterations
        self.current_iteration = 0

        phi = c1 + c2
        if phi <= 4:
            raise ValueError("c1 + c2 must be > 4 for constriction factor PSO.")
        kappa = 1.0
        self.chi = 2 * kappa / abs(2 - phi - np.sqrt(phi ** 2 - 4 * phi))

        self.gbest_position = None
        self.gbest_score    = float("inf")

        self.particles = [
            Particle(num_intersections, min_green, max_green)
            for _ in range(num_particles)
        ]

    def update_gbest(self, scores):
        for particle, score in zip(self.particles, scores):
            if score < particle.pbest_score:
                particle.pbest_score    = score
                particle.pbest_position = particle.position.copy()
            if score < self.gbest_score:
                self.gbest_score    = score
                self.gbest_position = particle.position.copy()

    def step(self, scores):
        self.update_gbest(scores)
        for particle in self.particles:
            r1 = np.random.random(self.num_intersections)
            r2 = np.random.random(self.num_intersections)
            cognitive = self.c1 * r1 * (particle.pbest_position - particle.position)
            social    = self.c2 * r2 * (self.gbest_position    - particle.position)
            particle.velocity = self.chi * (particle.velocity + cognitive + social)
            particle.position = np.clip(
                particle.position + particle.velocity,
                self.min_green,
                self.max_green,
            )
        self.current_iteration += 1

    def get_positions(self):
        return [p.position for p in self.particles]


class ConstrictionPSOOptimizer(PSOOptimizer):
    """PSOOptimizer subclass that uses ConstrictionSwarm instead of Swarm."""

    def __init__(self, fitness_evaluator, **kwargs):
        # Skip parent __init__ Swarm creation; build our own
        self.fitness_evaluator = fitness_evaluator
        self.max_iterations = kwargs["max_iterations"]
        self.history: list[float] = []

        if kwargs.get("seed") is not None:
            np.random.seed(kwargs["seed"])

        self.swarm = ConstrictionSwarm(
            num_particles     = kwargs["num_particles"],
            num_intersections = kwargs["num_intersections"],
            min_green         = kwargs["min_green"],
            max_green         = kwargs["max_green"],
            c1                = kwargs["c1"],
            c2                = kwargs["c2"],
            max_iterations    = kwargs["max_iterations"],
        )


def _build_constriction_optimizer(fitness_evaluator, kwargs):
    return ConstrictionPSOOptimizer(
        fitness_evaluator=fitness_evaluator,
        **{k: v for k, v in kwargs.items() if k != "inertia_strategy"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (run this file directly: python -m src.pso.variants)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ev = FitnessEvaluator(steps=100)
    for variant_name in VARIANTS:
        opt = build_variant(variant_name, fitness_evaluator=ev, seed=42,
                            num_particles=10, max_iterations=20)
        result = opt.run()
        print(f"{variant_name:25s}  score={result['best_score']:.0f}  "
              f"timings={result['best_position'].round(1)}")