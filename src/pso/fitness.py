from src.simulation.network_builder import build_network

class FitnessEvaluator:
    def __init__(self, steps=100):
        self.steps = steps
    
    def evaluate(self, position):
        # بنبني الشبكة من جديد في كل مرة عشان تبدأ نظيفة
        network = build_network()
        
        # بنشغل المحاكاة بالتوقيتات دي
        score = network.simulate(position, steps=self.steps)
        
        return score