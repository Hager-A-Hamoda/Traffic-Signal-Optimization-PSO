from src.simulation.network_builder import build_network
from src.pso.particle import Particle
from src.pso.fitness import FitnessEvaluator
from src.pso.optimizer import PSOOptimizer
from src.experiments.runner import ExperimentRunner

def main():
    network = build_network()

    timings = [20, 20, 20]

    network.reset()
    result = network.simulate(timings)

    print("\n Simulation Result")
    print("\n Timings:", timings)
    print("\n Total Waiting Time:", result, "vehicle-time")

    print("\n-------------------")
    evaluator = FitnessEvaluator()
    particle = Particle(num_intersections=3)
    score = evaluator.evaluate(particle.position)
    print(f"\n Timings: {particle.position}")
    print(f"\n Score: {score}")
    print("\n--- Running PSO Optimization ---")
    opt = PSOOptimizer(
        fitness_evaluator=evaluator,
        num_intersections=3,
        num_particles=20,
        max_iterations=50,
        seed=42
    )
    opt.swarm.diversity = "crowding" 
    result = opt.run()

    print(f"\n Best Timings:  {result['best_position'].round(2)}")
    print(f" Best Score:   {result['best_score']:.0f}")
    print(f" Improvement:    {((26701 - result['best_score']) / 26701 * 100):.1f}%")   

    print("\n--- Running Experiments (30 runs per variant) ---")
    runner = ExperimentRunner(num_runs=30)
    runner.run_all()
    runner.save_results("results/experiments.json")
    runner.save_csv("results/experiments.csv")
    runner.save_seeds("results/seeds.json")

if __name__ == "__main__":
    main()

