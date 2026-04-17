class TrafficNetwork:
    def __init__(self, intersections):
        self.intersections = intersections

    def reset(self):
        for i in self.intersections:
            i.queue = 0
            i.traffic_light.current_time = 0

    def simulate(self, timings, steps=100):
        for i, t in enumerate(timings):
            self.intersections[i].traffic_light.green_time = t

        total_waiting_time = 0

        for _ in range(steps):
            for inter in self.intersections:
                inter.step()

            total_waiting_time += sum(i.get_queue() for i in self.intersections)

        return total_waiting_time