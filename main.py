from src.simulation.network_builder import build_network
from src.pso.particle import Particle
from src.pso.fitness import FitnessEvaluator

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
if __name__ == "__main__":
    main()

