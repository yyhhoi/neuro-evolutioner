


class Probe():
    def __init__(self, simenv, neuron):
        self.neuron = neuron
        self.simenv = simenv
        self.potentials = []
        self.times = []
    def probe_potential(self):
        self.potentials.append(self.neuron.u)
    def probe_time(self):
        self.times.append(self.simenv.getTime())


    def get_potentials(self):
        return self.potentials
    def get_times(self):
        return self.times

