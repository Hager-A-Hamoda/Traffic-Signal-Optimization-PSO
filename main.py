from src.simulation.network_builder import build_network

def main():
    network = build_network()

    timings = [20, 20, 20]

    network.reset()
    result = network.simulate(timings)

    print("\n Simulation Result")
    print("\n Timings:", timings)
    print("\n Total Waiting Time:", result, "vehicle-time")

if __name__ == "__main__":
    main()