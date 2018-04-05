###########################
#  Fichier classe.py      #
# 04/04/18                #
# la communauté de l'info #
###########################

import numpy as np
import random
import time

class HMM:
    """ Define an HMM"""

    #mettre des raise...
    #faire des setter

    #virer les Nones
    def __init__(self, letters_number, states_number, initial, transitions, emissions):
        # The number of letters
        self.__letters_number = letters_number
        # The number of states
        self.__states_number = states_number
        # The vector defining the initial weights
        self.__initial = initial
        # The array defining the transitions
        self.__transitions = transitions
        # The list of vectors defining the emissions
        self.__emissions = emissions

    @property
    def letters_number(self):
        return self.__letters_number

    @property
    def states_number(self):
        return self.__states_number

    @property
    def initial(self):
        return self.__initial

    @property
    def transitions(self):
        return self.__transitions

    @property
    def emissions(self):
        return self.__emissions

    @initial.setter
    def initial(self, values):
        self.check_initial(values)
        self.__inital = values

    @staticmethod
    def check_initial(value):
        HMM.check_probability_array(value)
        if value.ndim != 1:
            raise ValueError("The parameter value should be a one dimension array")


    @staticmethod
    def check_probability_array(array):

        if not isinstance(array, np.ndarray):
            raise ValueError("The parameter array should be a np.ndarray")

        if array.ndim == 1:
            sum = 0
            for value in array:
                if value < 0:
                    raise ValueError("The probabilities should be positive numbers")
                sum += value

            if not np.isclose([sum], [1]):
                raise ValueError("The probabilities sum should be equal to 1 for each lign")

        elif array.ndim == 2:
            for i in range(array.shape[0]):
                sum = 0
                for j in range(array.shape[1]):
                    if array[i, j] < 0:
                        raise ValueError("The probabilities should be positive numbers")
                    sum += array[i, j]
                if not np.isclose([sum], [1]):
                    raise ValueError("The probabilities sum should be equal to 1 for each lign")

        else:
            raise ValueError("The dimension of the parameter array should be 1 or 2")

    def check_transition(self, value):
        self.check_dim(value, self.states_number, self.states_number)

    def check_emissions(self, value):
        self.check_dim(value, self.states_number, self.letters_number)

    @staticmethod
    def check_dim(tableau, nb_lignes, nb_colonnes):
        if tableau.ndim != 2:
            raise ValueError("The parameter tableau should be a 2D array")
        if tableau.shape[0] != nb_lignes or tableau.shape[1] != nb_colonnes:
            raise ValueError("Le tableau est de mauvaises dimensions")

    @transitions.setter
    def transitions(self, value):
        self.check_transition(value)
        self.__transitions = value

    @emissions.setter
    def emissions(self, value):
        self.check_emissions(value)
        self.__emissions = value

    @staticmethod
    def load(adr):
        """charge l'adresse"""

        data = open(adr, 'r')
        line = data.readline()
        hash_count = 0


        while hash_count <= 4:
            if line[0] == '#':
                if hash_count == 0 :
                    letters_number = int(data.readline())

                if hash_count == 1 :
                    states_number = int(data.readline())

                if hash_count == 2 :
                    initial = np.zeros((states_number))
                    for i in range(states_number):
                        initial[i] = float(data.readline())

                if hash_count == 3 :
                    transitions = np.zeros((states_number, states_number))
                    for i in range(states_number):
                        ligne = data.readline().split()
                        for j in range (len(ligne)):
                            transitions[i, j] = float(ligne[j])

                if hash_count == 4 :
                    emissions = np.zeros((states_number, letters_number))
                    for i in range(states_number):
                        ligne = data.readline().split()
                        for j in range(len(ligne)):
                            emissions[i, j] = float(ligne[j])

                hash_count += 1

            line = data.readline()

        data.close()

        return HMM(letters_number, states_number, initial, transitions, emissions)


    def __str__(self):
        return 'The number of letters : ' +  str(self.__letters_number) +'\n' +  ' The number of states : '+ str(self.__states_number) +'\n'+ ' The initial vector : ' +  str(self.__initial) +'\n'+  ' The internal transitions : ' +'\n'+ str(self.__transitions) +'\n'+ ' The emissions : ' +'\n'+ str(self.__emissions)

    @staticmethod
    def draw_multinomial(array):
        if array.ndim != 1:
            raise ValueError("The parameter array should be a 1D array")
        HMM.check_probability_array(array)

        random.seed(time.clock())
        random_number = random.random()
        probability_sum = 0
        for i, probability in enumerate(list):
            probability_sum += probability
            if random_number <= probability_sum:
                return i
        return array.shape[1] - 1

    def generate_random(self, n):
        sequence = np.zeros(n)
        actual_state = self.draw_multinomial(self.initial)
        for i in range(n):
            sequence[i] = self.draw_multinomial(self.emissions[actual_state])
            actual_state = self.draw_multinomial(self.transitions[actual_state])

    def save(self, address):
        nfile = open(address, "w")
        nfile.write("# The number of letters\n")
        nfile.write(str(self.letters_number) + "\n")
        nfile.write("# The number of states\n")
        nfile.write(str(self.states_number) + "\n")
        nfile.write("# The initial transitions\n")
        for p in self.initial:
            nfile.write(str(p) + "\n")
        nfile.write("# The internal transitions\n")
        for i in range(len(self.transitions)):
            for j in range(len(self.transitions[0])):
                nfile.write(str(self.transitions[i, j]) + " ")
            nfile.write("\n")
        nfile.write("# The emissions" + "\n")
        for i in range(len(self.emissions)):
            for j in range(len(self.emissions[0])):
                nfile.write(str(self.emissions[i, j]) + " ")
            nfile.write("\n")

        nfile.close()

    def __eq__(self, hmm2):
        if self.letters_number != hmm2.letters_number:
            return False
        if self.states_number != hmm2.states_number:
            return False
        if not np.isclose(self.initial,hmm2.initial):
            return False
        if not np.isclose(self.transitions, hmm2.transitions):
            return False
        if not np.isclose(self.emissions, hmm2.emissions):
            return False
        return True

'''
hmm = HMM(2, 2, np.array([0.5, 0.5]), np.array([[1, 1], [1, 1]]), np.array([[1, 1], [1, 1]]))

hmm.save("test1.txt")

HMM = HMM('test1.txt')
HMM.affiche()'''

#hmm1=HMM.load('test2.txt')

#print(hmm1)