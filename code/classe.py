
class HMM():
    """ Define an HMM"""

    def __init__(self, letters_number, states_number, initial, transitions, emissions):
        # The number of letters
        self.letters_number = letters_number
        # The number of states
        self.states_number = states_number
        # The vector defining the initial weights
        self.initial = initial
        # The array defining the transitions
        self.transitions = transitions
        # The list of vectors defining the emissions
        self.emissions = emissions

    def get_letters_number(self):
        return self.letters_number

    def get_states_number(self):
        return self.states_number

    def get_initial(self):
        return self.initial

    def get_transitions(self):
        return self.transitions

    def get_emissions(self):
        return self.emissions
