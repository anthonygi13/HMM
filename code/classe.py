
class HMM():
    """ Define an HMM"""

    def __init__(self, nbL, nbS, initial, transitions, emissions):
        # The number of letters
        self.nbL = nbL
        # The number of states
        self.nbS = nbS
        # The vector defining the initial weights
        self.initial = initial
        # The array defining the transitions
        self.transitions = transitions
        # The list of vectors defining the emissions
        self.emissions = emissions

    def get_nbL(self):
        return self.nbL

    def get_nbS(self):
        return self.nbS

    def get_initial(self):
        return self.initial

    def get_transitions(self):
        return self.transitions

    def get_emissions(self):
        return self.emissions