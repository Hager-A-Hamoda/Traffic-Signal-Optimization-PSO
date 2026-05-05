import numpy as np

class Intersection:
    import numpy as np

class Intersection:
    def __init__(self, traffic_light, lanes=1, base_arrival=2):
        self.traffic_light = traffic_light
        self.lanes = lanes
        self.base_arrival = base_arrival

        self.queue_main = 0
        self.queue_side = 0

        self.main_incoming = []
        self.side_incoming = []

        self.main_outgoing = []
        self.side_outgoing = []

        self.time = 0

    # Connections 
    def add_main_incoming(self, road):
        self.main_incoming.append(road)

    def add_side_incoming(self, road):
        self.side_incoming.append(road)

    def add_main_outgoing(self, road):
        self.main_outgoing.append(road)

    def add_side_outgoing(self, road):
        self.side_outgoing.append(road)

    # Arrival 
    def get_arrival_rate(self):
        # Rush hour
        if 30 <= self.time <= 70:
            return self.base_arrival * 2  # crowded
        else:
            return self.base_arrival  # ordinary

    # Simulation Step 
    def step(self):
        self.time += 1

        # arrivals from network
        incoming_main = sum(r.release() for r in self.main_incoming)
        incoming_side = sum(r.release() for r in self.side_incoming)

        # arrivals randomly (traffic generation)
        rand_main = np.random.poisson(self.get_arrival_rate())
        rand_side = np.random.poisson(self.get_arrival_rate() * 0.7)

        self.queue_main += incoming_main + rand_main
        self.queue_side += incoming_side + rand_side

        phase = self.traffic_light.get_phase()
        capacity = self.lanes * 2

        if phase == "main":
            passed = min(self.queue_main, capacity)
            self.queue_main -= passed

            to_main = int(passed * 0.7)
            to_side = passed - to_main

            for r in self.main_outgoing:
                r.add_cars(to_main // max(1, len(self.main_outgoing)))

        else:  # side phase
            passed = min(self.queue_side, capacity)
            self.queue_side -= passed
        
            for r in self.side_outgoing:
                r.add_cars(to_side // max(1, len(self.side_outgoing)))

        self.traffic_light.step()

    def get_queue(self):
        return self.queue_main + self.queue_side