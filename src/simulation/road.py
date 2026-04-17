class Road:
    def __init__(self, delay=3):
        self.delay = delay
        self.buffer = [[] for _ in range(delay)]

    def add_cars(self, num):
        self.buffer[0].append(num)

    def release(self):
        arrived = sum(self.buffer[-1])

        for i in range(self.delay - 1, 0, -1):
            self.buffer[i] = self.buffer[i - 1]

        self.buffer[0] = []

        return arrived