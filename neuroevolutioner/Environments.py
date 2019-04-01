

class Simulation():
    def __init__(self, stop_time = None, epsilon = 0.001):
        self.time = 0
        self.epsilon = epsilon
        self.stop_time = stop_time
        self.sim_stop = False

    def getTime(self):
        return self.time

    def increment(self):
        self.time += self.epsilon
        if self.time >= self.stop_time:
            self.sim_stop = True
        else:
            self.sim_stop = False    


