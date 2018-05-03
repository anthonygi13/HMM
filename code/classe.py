###########################
# Fichier classe.py       #
# 04/04/18                #
# La communauté de l'info #
###########################

# faire un assert que les lettres du mots w c bien dans les observables
# assert du type de w aussi

import numpy as np
import random
import time
import copy


def list_rand_sum_2_dim(n, m):
    '''creer un numpy aléatoire de dim n*m avec somme =1 sur toutes les lignes'''
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
            raise ValueError("The states number should be a positive integer")
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
        self.__initial = values

    @staticmethod
    def check_initial(value):
        HMM.check_probability_array(value)
        if value.ndim != 1:
            raise ValueError("The parameter value should be a one dimension array")

    @staticmethod
    def check_probability_array(array):
        '''verifie si la somme des valeurs sur une colonne est ==1 et qu'il n'y est pas de valeurs négatives'''

        if not isinstance(array, np.ndarray):
            raise TypeError("The parameter array should be a np.ndarray")

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


    def check_w(self, w):
        if type(w) != tuple:
            raise TypeError("w doit etre un tuple")

        for x in w:
            if type(x) != int:
                raise ValueError('les etats doivent etre des entiers')
            if x >= self.letters_number:
                raise ValueError("tout les elements doivent appartenir aux observables")

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
        '''return à partir dune liste de proba, l'indice avec la proba correspondant à sa valeur'''
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
        sequence = ()
        actual_state = self.draw_multinomial(self.initial)
        for i in range(n):
            sequence  += (self.draw_multinomial(self.emissions[actual_state]),)
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
        '''fonction forward'''
        #check w
        #marche
        if len(w) == 0:
            raise ValueError("w ne doit pas être vide")
        f = self.initial * self.emissions[:, w[0]]
        for i in range(1, len(w)):
            f = np.dot(f, self.transitions) * self.emissions[:, w[i]]
        return np.sum(f)


    def pbw(self,w):
        '''fonction forward'''
        #check w
        #marche
        if len(w) == 0:
            raise ValueError("w ne doit pas être vide")
        b = np.array([1]*self.states_number)
        for i in range(len(w)-2, -1, -1):
            b = np.dot(self.transitions, self.emissions[:, w[i+1]] * b)
        return np.sum(self.initial * b * self.emissions[:,w[0]])


    def predit(self, w):
        '''predit l'etat suivant'''
        #check w
        h = self.initial
        for i in range(1, len(w)):
            h = np.dot(self.emissions[:, w[i]] * h, self.transitions)
        p = np.dot(h, self.emissions)
        return np.argmax(p)


    def viterbi(self, w):
        #check w
        chemin_1 = []
        chemin_2 = []
        liste_etats = []
        p_1 = self.initial * self.emissions[:,w[0]]
        p_2 = self.initial * self.emissions[:,w[0]]
        for i in range(self.states_number):
            chemin_1 += [[i]]
            chemin_2 += [[i]]
            liste_etats += [i]

        for i in range(1,len(w)):
            for k in range (self.states_number):

                m = 0
                j_retenu = 0
                for j in range(self.states_number):
                    a = m
                    b = p_1[j] * self.transitions[j, k]
                    m = max(a,b)
                    if m == b :
                        j_retenu = j
                chemin_2[k] = chemin_1[j_retenu] + [k]
                p_2[k] = m * self.emissions[k, w[i]]
            chemin_1 = copy.deepcopy(chemin_2)
            p_1 = copy.deepcopy(p_2)
        print('p',p_2)
        return chemin_2[np.argmax(p_2)], np.log(np.max(p_2))


    def f(self, w): # verifier type de w
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
        return (f * b) / np.einsum('kt,kt->t', b, f)

    def xi(self,w):
        f = self.f(w)[:, :-1]
        b = self.b(w)[:, 1:]
        emissions = self.emissions[:, w[1:]]
        xi = np.einsum('kt,kl,lt,lt->klt', f, self.transitions, emissions, b)
        v = np.einsum('kt,kl,lt,lt->t', f, self.transitions, emissions, b)
        somme = np.tile(v, (self.states_number, self.states_number, 1))
        xi = xi / somme
        return xi

    def xi2(self,w):
        f = self.f(w)[:, :-1]
        b = self.b(w)[:, 1:]
        emissions = self.emissions[:, w[1:]]
        xi = np.einsum('kt,kl,lt,lt->klt', f, self.transitions, emissions, b)
        for t in range (xi.shape[2]):
            xi[:,:,t] = xi[:,:,t]/np.sum(xi[:,:,t])
        return xi


    def bw1(self, S):
        assert len(S) != 0

        pi = np.zeros(self.states_number)
        for j in range (len(S)):
            pi += np.array(self.gamma(S[j])[:, 0])


        T = np.zeros((self.states_number, self.states_number))
        for j in range (len(S)):
            for t in range (len(S[j]) - 1):
                T += self.xi(S[j])[:,:,t]

        O = np.zeros((self.states_number, self.letters_number))
        for j in range (len(S)):
            gamma = self.gamma(S[j])
            for t in range (len(S[j])):
                O[:, S[j][t]] += gamma[:,t]

        somme = pi.sum()
        self.initial = pi / somme

        somme = T.sum(1)
        self.transitions = (T.T / somme).T

        somme = O.sum(1)
        self.emissions = (O.T / somme).T



    def bw2(self, nbS, nbL, S, N):
        hmm = self.gen_HMM(nbL, nbS)
        for i in range (N):
            hmm.bw1(S)
        return hmm


    @staticmethod
    def bw3(self,nbS, nbL, S, N, M):
        max_logV = 0
        hmm = None
        for i in range (M):
            h = self.bw2(nbS, nbL, S, N)
            logV = hmm.logV(S)
            if max_logV < logV:
                max_logV = logV
                hmm = h
        return hmm


    def bw2_variante(self,nbS, nbL, S, limite, N=None):
        hmm = self.gen_HMM(nbL, nbS)
        logV  = hmm.logV(S)
        compteur = 0
        c = 0
        while compteur <= 10:
            if c != N :
                break

            hmm.bw1(S)
            logV_new = hmm.logV(S)

            if logV - logV_new < limite:
                compteur += 1
            else:
                compteur = 0

            logV = logV_new

            if N is not None :
                c += 1

        return hmm

    def bw3_variante(self,nbS, nbL, S, M, limite, N=None):
        max_logV = 0
        hmm = None
        for i in range (M):
            h = self.bw2_variante(nbS, nbL, S, limite, N)
            logV = hmm.logV(S)
            if max_logV < logV:
                max_logV = logV
                hmm = h
        return hmm


    @staticmethod
    def gen_HMM(nbr_lettre, nbr_etat): #faire des checks sur les parametres
        '''genere un HMM aléatoire avec un nombre de lettre et d'etat en entree'''
        letters_number = nbr_lettre
        states_number = nbr_etat
        initial = list_rand_sum_2_dim(1, nbr_etat)
        transitions = list_rand_sum_2_dim(nbr_etat, nbr_etat)
        emissions = list_rand_sum_2_dim(nbr_etat, nbr_lettre)

        return HMM(letters_number, states_number, initial[0], transitions, emissions)



    def logV(self, S):
        # il est ou le log ?
        somme = 0
        for w in S:
            somme += np.log(self.pfw(w))
        return somme

    @staticmethod
    def num_to_lettre(n):
        if type(n) != int:
            raise TypeError('le numero doit etre un entier')
        if n < 0 or n > 25:
            raise ValueError('le numero doit etre compris entre 0 et 25')

        alphabet =['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

        return alphabet[n]


    @staticmethod
    def lettre_to_num(lettre):
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                    'u', 'v', 'w', 'x', 'y', 'z']
        if type(lettre) != str:
            raise TypeError('la lettre doit etre un caractere')
        if lettre not in alphabet:
            raise ValueError('la lettre doit etre dans l\'alphabet' )

        return alphabet.index(lettre)




A = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print((B.T/np.array([2, 4, 8])).T)

#print(A)
#print(B)
#print(A * B)
H = HMM.load('test1.txt')
H.bw1([(0,1)])
print (H.transitions)
C = [1,2,3]
D = copy.deepcopy(C)
C.append(2)
print(D)