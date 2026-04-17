import numpy as np

class Intersection:
    def __init__(self, traffic_light, lanes=1, base_arrival=2):
        self.traffic_light = traffic_light
        self.lanes = lanes
        self.base_arrival = base_arrival
        self.queue = 0

        self.incoming_roads = []
        self.outgoing_roads = []

        self.time = 0  # tracking time 

    def add_incoming(self, road):
        self.incoming_roads.append(road)

    def add_outgoing(self, road):
        self.outgoing_roads.append(road)

    def get_arrival_rate(self):
        # Rush hour
        if 30 <= self.time <= 70:
            return self.base_arrival * 2  # crowded
        else:
            return self.base_arrival  # ordinary

    def step(self):
        self.time += 1

        arrival_rate = self.get_arrival_rate()
        incoming = np.random.poisson(arrival_rate)

        incoming_from_roads = 0
        for road in self.incoming_roads:
            incoming_from_roads += road.release()

        self.queue += incoming + incoming_from_roads

        if self.traffic_light.is_green():
            capacity = self.lanes * 2
            passed = min(self.queue, capacity)
            self.queue -= passed

            if self.outgoing_roads:
                per_road = passed // len(self.outgoing_roads)
                for road in self.outgoing_roads:
                    road.add_cars(per_road)

        self.traffic_light.step()

    def get_queue(self):
        return self.queue