from src.simulation.network_builder import build_network
from src.pso.particle import Particle
from src.pso.fitness import FitnessEvaluator
from src.pso.optimizer import PSOOptimizer
from src.DE.de import DEOptimizer
from src.experiments.runner import ExperimentRunner
from src.hypered.hyp import HybridPSODE

def main():
    evaluator = FitnessEvaluator()
    SEED = 22

    # DE
    print("\n--- Running DE Optimization ---")
    de = DEOptimizer(
        fitness_evaluator=evaluator,
        num_intersections=3,
        population_size=30,
        max_iterations=50,
        representation="continuous",
        init_strategy="random",
        mutation_strategy="rand1",
        crossover_strategy="binomial",
        parent_selection="random",
        seed=SEED
    )
    de_result = de.run()
    print(f"\n Best Timings:  {de_result['best_position'].round(2)}")
    print(f" Best Score:    {de_result['best_score']:.0f}")

    # PSO
    print("\n--- Running PSO Optimization ---")
    opt = PSOOptimizer(
        fitness_evaluator=evaluator,
        num_intersections=3,
        num_particles=30,
        max_iterations=50,
        seed=SEED
    )
    pso_result = opt.run()
    print(f"\n Best Timings:  {pso_result['best_position'].round(2)}")
    print(f" Best Score:    {pso_result['best_score']:.0f}")

    # المقارنة
    print("\n--- Comparison ---")
    if de_result['best_score'] < pso_result['best_score']:
        print(" DE is better")
    else:
        print(" PSO is better")

    print("\n--- Running Experiments ---")
    runner = ExperimentRunner(num_runs=30)
    runner.run_all()
    
    hybrid = HybridPSODE(
        fitness_evaluator=evaluator,
        num_intersections=3,
        num_particles=30,
        max_iterations=100,
        F=0.6,
        CR=0.8,
        w=0.4,
        seed=SEED
    )
    hybrid_result = hybrid.run()
    print(f"\n Best Timings:  {hybrid_result['best_position'].round(2)}")
    print(f" Best Score:    {hybrid_result['best_score']:.0f}")

    # المقارنة النهائية
    print("\n--- Final Comparison ---")
    print(f" PSO Score:    {pso_result['best_score']:.0f}")
    print(f" DE Score:     {de_result['best_score']:.0f}")
    print(f" Hybrid Score: {hybrid_result['best_score']:.0f}")

    scores = {
        "PSO": pso_result['best_score'],
        "DE": de_result['best_score'],
        "Hybrid": hybrid_result['best_score']
    }
    best = min(scores, key=scores.get)
    print(f"\n Best Algorithm: {best}")

    runner.save_results("results/experiments.json")
    runner.save_csv("results/experiments.csv")
    runner.save_seeds("results/seeds.json")

if __name__ == "__main__":
    main()