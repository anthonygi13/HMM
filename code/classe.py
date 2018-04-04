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

        data = open(adr, 'r')
        line = data.readline()
        c = 0


        while c!= 5:
            if line[0] == '#':
                if c == 0 :
                    self.__letters_number = int(data.readline())

                if c == 1 :
                    self.__states_number = int(data.readline())

                if c == 2 :
                    self.__initial = np.zeros((1, self.__states_number))
                    for i in range(self.__states_number):
                        self.__initial[0][i] = int(data.readline())

                if c == 3 :
                    self.__transitions = np.zeros((self.__states_number, self.__states_number))
                    for i in range(self.__letters_number):
                        self.__transitions = np.array([int(data.readline()), int(data.readline())])

                if c == 4 :
                    for i in range(self.__letters_number):
                        self.__emissions = np.array([int(data.readline()), int(data.readline())])

                c += 1


            line = data.readline()


        data.close()


    def gen_rand(self,n):
        initial_additionne = np.zeros(1, n)
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

np.array