###########################
# Fichier classe.py       #
# 04/04/18                #
# La communauté de l'info #
###########################

# faire un assert que les lettres du mots w c bien dans les observables

import numpy as np
import random
import time




def list_rand_sum_2_dim(n, m):
    if type(n) != int:
        raise ValueError('le nombre de lettre doit etre un entier')
    if type(m) != int:
        raise ValueError('le nombre etat doit etre un entier')

    L = np.zeros((n, m-1))

    for j in range(n): #j colonne
        for i in range(m-1): #i sur la ligne
            s = random.random()
            s = "%.3f" % s
            L[j, i] = (s)

    for i in range(n):
        L.sort()

    M = np.zeros((n, m))



    for i in range(n): #pour chaque ligne
        for j in range(0, m): #pour chaque colonne
            if j == m-1:
                M[i, j] = 1 - L[i, j-1]
            elif j == 0:
                M[i, j] = L[i, j]
            else :
                M[i,j] = L[i, j] - L[i, j-1]


    return M


class HMM:
    """ Define an HMM"""

    # mettre des raise au niveau du __init__
    # faire des setter

    # virer les Nones
    def __init__(self, letters_number, states_number, initial, transitions, emissions):
        # The number of letters
        if type(letters_number) != int or letters_number <= 0:
            raise ValueError("The letters number should be a positive integer")
        self.__letters_number = letters_number
        # The number of states
        if type(states_number) != int or states_number <= 0:
            if type(self.letters_number) != int:
                raise ValueError("The letters number should be a positive integer")
        self.__states_number = states_number
        # The vector defining the initial weights
        self.check_initial(initial)
        self.__initial = initial
        # The array defining the transitions
        self.check_transitions(transitions)
        self.__transitions = transitions
        # The list of vectors defining the emissions
        self.check_emissions(emissions)
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
        '''verifie si la somme des valeurs sur une colonne est ==1 et qu'il n'y est pas de valeurs négatives'''

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

    def check_transitions(self, value):
        # verifie si la somme des valeurs sur une colonne est = 1 et qu'il n'y est pas de valeurs négatives dans tableau de transition
        self.check_dim(value, self.states_number, self.states_number)
        self.check_probability_array(value)

    def check_emissions(self, value):
        # verifie si la somme des valeurs sur une colonne est = 1 et qu'il n'y est pas de valeurs négatives dans tableau de emission
        self.check_dim(value, self.states_number, self.letters_number)
        self.check_probability_array(value)

    @staticmethod
    def check_dim(tableau, nb_lignes, nb_colonnes):
        if tableau.ndim != 2:
            raise ValueError("The parameter tableau should be a 2D array")
        if tableau.shape[0] != nb_lignes or tableau.shape[1] != nb_colonnes:
            raise ValueError("Le tableau est de mauvaises dimensions")

    @transitions.setter
    def transitions(self, value):
        self.check_transitions(value)
        self.__transitions = value

    @emissions.setter
    def emissions(self, value):
        self.check_emissions(value)
        self.__emissions = value

    @staticmethod
    def load(adr):
        """charge un HMM depuis une adresse donnee"""

        data = open(adr, 'r')
        line = data.readline()
        hash_count = 0

        while hash_count <= 4:
            if line[0] == '#':
                if hash_count == 0:
                    letters_number = int(data.readline())

                if hash_count == 1:
                    states_number = int(data.readline())

                if hash_count == 2:
                    initial = np.zeros((states_number))
                    for i in range(states_number):
                        initial[i] = float(data.readline())

                if hash_count == 3:
                    transitions = np.zeros((states_number, states_number))
                    for i in range(states_number):
                        ligne = data.readline().split()
                        for j in range(len(ligne)):
                            transitions[i, j] = float(ligne[j])

                if hash_count == 4:
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
        return 'The number of letters : ' + str(self.__letters_number) + '\n' + ' The number of states : ' + str(
            self.__states_number) + '\n' + ' The initial vector : ' + str(
            self.__initial) + '\n' + ' The internal transitions : ' + '\n' + str(
            self.__transitions) + '\n' + ' The emissions : ' + '\n' + str(self.__emissions)

    @staticmethod
    def draw_multinomial(array):
        if array.ndim != 1:
            raise ValueError("The parameter array should be a 1D array")
        HMM.check_probability_array(array)

        random.seed(time.clock())
        random_number = random.random()
        probability_sum = 0
        for i, probability in enumerate(array):
            probability_sum += probability
            if random_number <= probability_sum:
                return i
        return array.shape[1] - 1

    def generate_random(self, n):
        if type(n) != int:
            raise ValueError("n doit être un entier")
        sequence = np.zeros(n)
        actual_state = self.draw_multinomial(self.initial)
        for i in range(n):
            sequence[i] = self.draw_multinomial(self.emissions[actual_state])
            actual_state = self.draw_multinomial(self.transitions[actual_state])
        return sequence

    def save(self, address):
        '''sauvegarde un HMM'''
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
        if not np.isclose(self.initial, hmm2.initial):
            return False
        if not np.isclose(self.transitions, hmm2.transitions):
            return False
        if not np.isclose(self.emissions, hmm2.emissions):
            return False
        return True

    def pfw(self, w):
        #check w
        #marche
        if len(w) == 0:
            raise ValueError("w ne doit pas être vide")
        f = self.initial * self.emissions[:, w[0]]
        print (f)
        for i in range(1, len(w)):
            f = np.dot(f, self.transitions) * self.emissions[:, w[i]]
        print(f)
        return np.sum(f)


    def pbw(self,w):
        #check w
        #marche
        if len(w) == 0:
            raise ValueError("w ne doit pas être vide")
        b = np.array([1]*self.states_number)
        for i in range(len(w)-2, -1, -1):
            b = np.dot(self.transitions, self.emissions[:, w[i+1]] * b)
        print(b)
        return np.sum(self.initial * b * self.emissions[:,w[0]])


    def predit(self, w):
        #check w
        h = self.initial
        for i in range(1, len(w)):
            h = np.dot(self.emissions[:, w[i]] * h, self.transitions)
        p = np.dot(h, self.emissions)
        return np.argmax(p)


    def viterbi(self, w):
        #check w
        chemin = []
        liste_etats = []
        p = self.initial * self.emissions[:,w[0]]
        for i in range(self.states_number):
            chemin += [[i]]
            liste_etats += [i]
        for i in range(len(w)):
            for k in range (len(liste_etats)):
                m = 0
                j_retenu = 0
                for j in range(len(p)):
                    a = m
                    b = p[j] * self.transitions(liste_etats[j], k)
                    m = max(a,b)
                    p[i] = m * self.emissions(k, w[i])
                    if m == b :
                        j_retenu = j
                chemin += chemin[j_retenu] + [k]
        return chemin[np.argmax(p)]


    def f(self, w):
        if len(w) == 0:
            raise ValueError("w ne doit pas être vide")
        f = np.zeros((self.states_number, len(w)))
        f[:, 0] = self.initial * self.emissions[:, w[0]]
        for i in range(1, len(w)):
            f[:, i] = np.dot(f[:, i-1], self.transitions) * self.emissions[:, w[i]]
        return f

    def b(self, w):
        # check w
        if len(w) == 0:
            raise ValueError("w ne doit pas être vide")
        b = np.zeros((self.states_number, len(w)))
        b[:, len(w)-1] = np.array([1]*self.states_number)
        for i in range (len(w)-2, -1, -1):
            b[:, i] = np.dot(self.transitions, self.emissions[:, w[i+1]] * b[:, i+1])
        return b

    def gamma(self, w):
        # check w
        f = self.f(w)
        b = self.b(w)
        return (f * b) / np.einsum('ji,ji->i', b, f)

    def xi(self,w):
        xi = np.ones((self.states_number, self.states_number, len(w)))
        f = self.f(w)
        b = self.b(w)
        for t in range (len(w)-1):
            xi[:, :, t] = np.dot(f[:,t] * self.transitions * self.emissions[:,w[t+1]], b[:,t+1])
            xi[:, :, t] = xi[:,:,t] / np.sum(xi[t])
        return xi


    def bw1(self, S):
        assert len(S) != 0
        somme1 = 0
        pi = np.zeros(self.states_number)
        for j in range (len(S)):
            pi += np.array(self.gamma(S[j])[:, 0])
        somme = pi.sum()
        self.__initial = pi/somme
#        for i in range (len(pi)):
 #           somme += pi[i]
  #      for k in range (len(pi)):
   #         pi[k] = pi[k] / somme
    #        self.initial[k] = pi[k]

        T = np.zeros((self.states_number, self.states_number))
        for j in range (len(S)):
            for t in range (len(S[j]) - 1):
                T += self.xi(S[j])[:,:,t]           #C'est bien comme ça qu'on fixe juste une variable et qu'on prend le reste ?
        somme = T.sum(1)
        for k in range (self.states_number):
            self.__transitions[k] = T[k]/somme[k]


        O = np.zeros((self.states_number, self.letters_number))
        for j in range (len(S)):
            for t in range (S[j]):
                for o in range (self.letters_number):
                    O[:, o] += self.gamma(S[j])[:, t]
        somme = O.sum(1)
        for k in range (self.states_number):
            self.__emissions[k] = O[k]/somme[k]

    def bw2(self, nbS, nbL, S, N):
        hmm = self.gen_HMM(nbL, nbS)
        for i in range (N):
            hmm = hmm.bw1(S)
        return hmm

    def bw3(self,nbS, nbL, w, N, M):
        for i in range (M):
            pass


#4.5.2 : faire bw sur la séquence de mots
#4.5.5 : predit ?



    @staticmethod
    def gen_HMM(nbr_lettre, nbr_etat): #faire des checks sur les parametres
        letters_number = nbr_lettre
        states_number = nbr_etat
        initial = list_rand_sum_2_dim(1, nbr_etat)
        # print(initial)
        transitions = list_rand_sum_2_dim(nbr_etat, nbr_etat)
        # print(transitions)
        emissions = list_rand_sum_2_dim(nbr_etat, nbr_lettre)
        # print(emissions)

        return HMM(letters_number, states_number, initial[0], transitions, emissions)



    def logV(self, S):
        somme = 0
        for w in S:
            somme += self.pfw(w)
        return somme



