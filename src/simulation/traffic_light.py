class TrafficLight:
    def __init__(self, green_time, red_time):
        self.green_time = green_time
        self.red_time = red_time
        self.current_time = 0

    def step(self):
        self.current_time += 1

    def is_green(self):
        cycle = self.green_time + self.red_time
        return (self.current_time % cycle) < self.green_time 