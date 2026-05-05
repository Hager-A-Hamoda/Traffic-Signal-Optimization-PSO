class TrafficNetwork:
    def __init__(self, intersections):
        self.intersections = intersections

    def reset(self):
        for i in self.intersections:
            i.queue_main = 0
            i.queue_side = 0
            i.time = 0
            i.traffic_light.current_time = 0

    def simulate(self, timings, steps=100):
        # set timings
        for i, t in enumerate(timings):
            tl = self.intersections[i].traffic_light
            cycle = tl.green_main + tl.green_side  # fixed
            t = max(1, min(t, cycle - 1))
            tl.green_main = t
            tl.green_side = cycle - t

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