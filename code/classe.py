######################
# Fichier class HMM  #
# 04/04/18           #
######################

import numpy as np

class HMM():
    """ Define an HMM"""

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
    

    def load(self, adr):
        """charge l'adresse"""
        pass
        """
        data = open(adr, 'r')
        line = data.readline()

        if line[0] == '#':
            self.__letters_number = int(data.readline())

        line = data.readline()
        if line[0] == '#':
            self.__states_number = int(data.readline())

        for i in range (self.__letters_number):
            self.__initial = np.array ([int(data.readline()), int(data.readline())])



            if column_number == n_c1:
                if char != " " and char != column_separator and char != "\n":
                    c1 += char
            if column_number == n_c2:
                if char != " " and char != column_separator and char != "\n":
                    c2 += char
            if column_number > max([n_c1, n_c2]):
                break
        if c1 == "":
            c1 = None
        if c2 == "":
            c2 = None
        for i in range(nb):
            if i == 2:
                d = l.split()
                s1 = int(d[0])  # sommet 1

            l = f.readline()

        f.close()
        """
    """def gen_rand(self,n):
        initial_additionne = []
        for i in range (len(self.initial)):
            if i==0:
     """

    def save(self, address):
        #faire en sorte que ça écrase bien avant d ecrire
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

hmm = HMM(2, 2, np.array([0.5, 0.5]), np.array([[1, 1], [1, 1]]), np.array([[1, 1], [1, 1]]))

hmm.save("test.txt")
