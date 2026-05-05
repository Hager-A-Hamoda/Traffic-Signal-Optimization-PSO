class TrafficNetwork:
    def __init__(self, intersections):
        self.intersections = intersections

    def reset(self):
        for i in self.intersections:
            i.queue = 0
            i.traffic_light.current_time = 0

    def simulate(self, timings, steps=100):
        # set timings
        for i, t in enumerate(timings):
            self.intersections[i].traffic_light.green_time = t

        total_waiting_time = 0
        queue_history = []  # per-step metrics

        for step in range(steps):
            for inter in self.intersections:
                inter.step()

            # total queue 
            total_queue = sum(i.get_queue() for i in self.intersections)

            total_waiting_time += total_queue
            queue_history.append(total_queue)

        # average waiting time
        avg_waiting_time = total_waiting_time / steps

        return {
            "total_waiting_time": total_waiting_time,
            "avg_waiting_time": avg_waiting_time,
            "queue_history": queue_history
        }