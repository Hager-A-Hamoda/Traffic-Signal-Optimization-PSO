class TrafficLight:
    def __init__(self, green_main, green_side):
        self.green_main = green_main
        self.green_side = green_side
        self.current_time = 0

    def step(self):
        self.current_time += 1

    def get_phase(self):
        cycle = self.green_main + self.green_side
        t = self.current_time % cycle

        if t < self.green_main:
            return "main"
        else:
            return "side"