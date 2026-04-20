import numpy as np

class Particle:
    def __init__(self, num_intersections, min_green=5, max_green=60):
        
        # التوقيتات الحالية (الحل)
        self.position = np.random.uniform(min_green, max_green, num_intersections)
        
        # السرعة (هتتغير بيها التوقيتات)
        self.velocity = np.random.uniform(-5, 5, num_intersections)
        
        # أحسن حل وصتله الجسيمة دي لوحدها
        self.pbest_position = self.position.copy()
        self.pbest_score = float('inf')

    def update_velocity(self, gbest_position, inertia=0.5, c1=1.5, c2=1.5):
        r1 = np.random.random(len(self.position))
        r2 = np.random.random(len(self.position))
        
        self.velocity = (inertia * self.velocity +
                        c1 * r1 * (self.pbest_position - self.position) +
                        c2 * r2 * (gbest_position - self.position))

    def update_position(self, min_green=5, max_green=60):
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, min_green, max_green)