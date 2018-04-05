######################
# Fichier classe.py  #
# 04/04/18           #
######################

import numpy as np
import random

class HMM():
    """ Define an HMM"""

    #mettre des raise...
    #faire des setter

    #virer les Nones
    def __init__(self, letters_number=None, states_number=None, initial=None, transitions=None, emissions=None):
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

    def check_initial(self,values):
        somme = 0
        for i in range (len(self.initial)):
            if self.initial[i] < 0:
                raise ValueError ("Toutes les probabilités doivent être positives")
            somme += self.initial[i]
        if not isclose(somme,1):
            raise ValueError ("La somme des probabilités doit être égale à 1")

    def check_tran

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

    @letters_number.getter
    def get_letters_number(self):
        return self.__letters_number

    @states_number.getter
    def get_letters_number(self):
        return self.__states_number

    @initial.setter
    def initial(self,valeur):



    """"
    @transitions.setter
    def transitions(self, values):
        values = self.check_transitions(values, self.__states_number)
        self.__transitions = values
    """



    def load(self, adr):
        """charge l'adresse"""

        data = open(adr, 'r')
        line = data.readline()
        c = 0 #donner des noms pas ambigus aux variables


        while c!= 5:
            if line[0] == '#':
                if c == 0 :
                    self.__letters_number = int(data.readline())

                if c == 1 :
                    self.__states_number = int(data.readline())

                if c == 2 :
                    self.__initial = np.zeros((1, self.__states_number))
                    for i in range(self.__states_number):
                        self.__initial[0][i] = float(data.readline())

                if c == 3 :
                    self.__transitions = np.zeros((self.__states_number, self.__states_number))
                    for i in range(self.__letters_number):
                        ligne = data.readline().split()
                        for j in range (len(ligne)):
                            self.__transitions[i][j] = float(ligne[j])

                if c == 4 :
                    self.__emissions = np.zeros((self.__states_number, self.__states_number))
                    for i in range(self.__letters_number):
                        ligne = data.readline().split()
                        for j in range(len(ligne)):
                            self.__emissions[i][j] = float(ligne[j])

                c += 1


            line = data.readline()


        data.close()

    def affiche(self):
        print('The number of letters : ', self.__letters_number)
        print('The number of states : ', self.__states_number)
        print('The initial transitions : ', self.__initial)
        print('The internal transitions : ', self.__transitions)
        print('The emissions : ', self.__emissions)



    def gen_rand(self,n):
        #long
        initial_additionne = np.zeros(len(self.initial), 1)
        for i in range (len(self.initial[0])):
            if i==0:
                initial_additionne[0,0] =  self.initial[0,0]
            else:
                initial_additionne[i,0] = self.initial[i,0] + self.initial[i - 1, 0]

        transition_additionne = np.zeros(len(self.transitions), len(self.transitions[0]))
        for i in range(len(self.transitions)):
            for j in range(len(self.transitions[0])):
                if j == 0:
                    transition_additionne[i,0] = self.transitions[i,0]
                else:
                    transition_additionne[i,j] = self.transitions[i,j] + self.transitions[i, j-1]

        emissions_additionne = np.zeros(len(self.emissions), len(self.emissions[0]))
        for i in range(len(self.emissions)):
            for j in range(len(self.emissions[0])):
                if j == 0:
                    emissions_additionne[i,0] = self.emissions[i,0]
                else:
                    emissions_additionne[i,j] = self.emissions[i,j] + self.emissions[i, j-1]

        nb = random.random()
        for i in range (len(self.initial)):
            p_initial = initial_additionne[i,1]
            if p_initial >= nb:
                val_initial = i
            break
        val_etat = val_initial
        res = self.__letters_number[val_initial]
        for var in range (n):
            for j in range(len(self.transitions[0])):
                p_transition = transition_additionne[val_etat, j]
                if p_transition >= nb:
                    val_transition = j
                break
            for k in range(len(self.emissions[0])):
                p_emission = emissions_additionne[val_etat, k]
                if p_emission >= nb:
                    val_emission = k
                break
            res += self.__letters_number[val_emission]
            val_etat = val_transition
        return res

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

hmm = HMM(2, 2, np.array([0.5, 0.5]), np.array([[1, 1], [1, 1]]), np.array([[1, 1], [1, 1]]))

hmm.save("test.txt")

#HMM = HMM('test.txt')
#HMM.affiche()