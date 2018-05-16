###########################
# Fichier classe.py       #
# 04/04/18                #
# La communauté de l'info #
###########################


import numpy as np
import random
import time
import copy


class HMM:
    # Défint un HMM

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
        """
        :param value: Vecteur d'initial
        :return: Renvoie une erreur si le tableau n'est pas de dimension 1, à une valeur négative ou à une somme de
        valeurs différente de 1.
        """
        HMM.check_probability_array(value)
        if value.ndim != 1:
            raise ValueError("The parameter value should be a one dimension array")

    @staticmethod
    def check_probability_array(array):
        """
        :param array: Tableau numpy
        :return: Renvoie une erreur si la somme des valeurs sur une ligne ne vaut pas 1 et si'il y a des valeurs
        négatives
        """
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
        """
        :param value: Matrice de transitions
        :return: Renvoie une erreur si la somme des valeurs sur une ligne ne vaut pas 1 et si'il y a des valeurs
        négatives
        """
        self.check_dim(value, self.states_number, self.states_number)
        self.check_probability_array(value)

    def check_emissions(self, value):
        """
        :param value: Matrice d'émissions
        :return: Renvoie une erreur si la somme des valeurs sur une ligne ne vaut pas 1 et si'il y a des valeurs
        négatives
        """
        self.check_dim(value, self.states_number, self.letters_number)
        self.check_probability_array(value)

    @staticmethod
    def check_dim(tableau, nb_lignes, nb_colonnes):
        """
        :param tableau: Tableau numpy
        :param nb_lignes: Nombre de lignes attendues
        :param nb_colonnes: Nombre de colonnes attendues
        :return: Renvoie une erreur si les dimensions du tableau ne sont pas celles attendues
        """
        if type(nb_lignes) != int:
            raise TypeError('nb_lignes doit etre un entier')
        if type(nb_colonnes) != int:
            raise TypeError('nb_colonnes doit etre un entier')
        if nb_lignes < 0:
            raise ValueError("nb_lignes doit être positif")
        if nb_colonnes < 0:
            raise ValueError("nb_colonnes doit être positif")
        if tableau.ndim != 2:
            raise ValueError("The parameter tableau should be a 2D array")
        if tableau.shape[0] != nb_lignes or tableau.shape[1] != nb_colonnes:
            raise ValueError("Le tableau est de mauvaises dimensions")

    def check_w(self, w):
        """
        :param w: Tuple
        :return: Renvoie une erreur si w n'est pas un tuple ou s'il contient des éléments qui ne sont pas des
        observables
        """
        if type(w) != tuple:
            raise TypeError("w doit etre un tuple")

        if len(w) == 0:
            raise ValueError("w ne doit pas être vide")

        for x in w:
            if type(x) != int:
                raise TypeError('Les lettres d\'un mot doivent etre des entiers')
            if x >= self.letters_number:
                print('x',x)
                print(self.letters_number)
                raise ValueError("Tous les éléments doivent appartenir aux observables")

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
        # Charge un HMM depuis une adresse donnee
        """
        :param adr: Nom d"un fichier texte contenant les données d'un HMM
        :return: Un HMM
        """
        if type(adr) != str:
            raise TypeError("adr doit être une chaîne de caractères")
        if adr == "":
            raise ValueError("adr ne doit pas être une chaîne de caractères vide")

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
        # Affiche le HMM
        """
        :return: Le HMM correspondant en affichant le nom des différentes entrées
        """
        return 'The number of letters : ' + str(self.__letters_number) + '\n' + ' The number of states : ' + str(
            self.__states_number) + '\n' + 'The initial vector : ' + str(
            self.__initial) + '\n' + 'The internal transitions : ' + '\n' + str(
            self.__transitions) + '\n' + 'The emissions : ' + '\n' + str(self.__emissions)

    @staticmethod
    def draw_multinomial(array):
        """
        :param array: Tableau de probabilités dont la somme des vleurs vaut 1
        :return: Un indice du tableau avec une probabilité égale à la valeur correspondant à l'indice
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("The parameter array should be a np.ndarray")
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
        """
        :param n: Longueur souhaitée de la liste
        :return: Un tuple d'oservables de longueur n où chaque observable est choisit avec une probabilité
        égale à sa probabilité correspondante
        """
        if type(n) != int:
            raise ValueError("n doit être un entier")
        if n < 0:
            raise ValueError("n doit être positif")
        sequence = ()
        actual_state = self.draw_multinomial(self.initial)
        for i in range(n):
            sequence += (self.draw_multinomial(self.emissions[actual_state]),)
            actual_state = self.draw_multinomial(self.transitions[actual_state])
        return sequence

    def save(self, address):
        """
        :param address: Nom du fichier
        :return: HMM sous fichier texte nommé par address
        """
        if type(address) != str:
            raise TypeError("adr doit être une chaîne de caractères")
        if address == "":
            raise ValueError("adr ne doit pas être une chaîne de caractères vide")

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
        # Définition de l'égalité entre 2 HMM
        if not isinstance(hmm2, HMM):
            raise TypeError("ne peut pas verifier une egalité entre un HMM et un objet qui n'est pas un HMM")
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
        """""
        :param w: tuple d'observables
        :return: Probabilité d'obtenir cette liste
        """
        self.check_w(w)
        f = self.initial * self.emissions[:, w[0]]
        for i in range(1, len(w)):
            f = np.dot(f, self.transitions) * self.emissions[:, w[i]]
        return np.sum(f)

    def pbw(self, w):
        """
        :param w: tuple d'observables
        :return: Probabilité d'obtenir cette liste
        """
        self.check_w(w)
        b = np.array([1] * self.states_number)
        for i in range(len(w) - 2, -1, -1):
            b = np.dot(self.transitions, self.emissions[:, w[i + 1]] * b)
        return np.sum(self.initial * b * self.emissions[:, w[0]])

    def predit(self, w):
        """
        :param w: tuple d'observables
        :return: L'observable ayant la plus grande probabilité d'apparaitre ensuite
        """
        self.check_w(w)
        h = self.initial
        for i in range(1, len(w)):
            h = np.dot(self.emissions[:, w[i]] * h, self.transitions)
        p = np.dot(h, self.emissions)
        return np.argmax(p)

    def viterbi(self, w):
        """
        :param w: tuple d'observables
        :return: La liste d'états la plus probable correspondant à ce chemin
        """
        self.check_w(w)
        chemin_1 = []
        chemin_2 = []
        liste_etats = []
        p_1 = self.initial * self.emissions[:, w[0]]
        p_2 = self.initial * self.emissions[:, w[0]]
        for i in range(self.states_number):
            chemin_1 += [[i]]
            chemin_2 += [[i]]
            liste_etats += [i]
        for i in range(1, len(w)):
            for k in range(self.states_number):
                m = 0
                j_retenu = 0
                for j in range(self.states_number):
                    a = m
                    b = p_1[j] * self.transitions[j, k]
                    m = max(a, b)
                    if m == b:
                        j_retenu = j
                chemin_2[k] = chemin_1[j_retenu] + [k]
                p_2[k] = m * self.emissions[k, w[i]]
            chemin_1 = copy.deepcopy(chemin_2)
            p_1 = copy.deepcopy(p_2)
        return chemin_2[np.argmax(p_2)], np.log(np.max(p_2))

    def f(self, w):
        """
        :param w: tuple d'observable
        :return: Matrice de dimension nb_d'etats * len(w) correspondant au f du polycopié 4.4
        """
        f = np.zeros((self.states_number, len(w)))
        f[:, 0] = self.initial * self.emissions[:, w[0]]
        for i in range(1, len(w)):
            f[:, i] = np.dot(f[:, i - 1], self.transitions) * self.emissions[:, w[i]]
        return f

    def b(self, w):
        """
        :param w: tuple d'observable
        :return: Matrice de dimension nb_d'etats * len(w) correspondant au b du polycopié 4.4
        """
        b = np.zeros((self.states_number, len(w)))
        b[:, len(w) - 1] = np.array([1] * self.states_number)
        for i in range(len(w) - 2, -1, -1):
            b[:, i] = np.dot(self.transitions, self.emissions[:, w[i + 1]] * b[:, i + 1])
        return b

    def gamma(self, w):
        """
        :param w: tuple d'observable
        :return: Matrice de dimension nb_d'etats * len(w) correspondant au gamma du polycopié 4.4
        """
        f = self.f(w)
        b = self.b(w)
        return (f * b) / np.einsum('kt,kt->t', b, f)

    def xi(self, w):
        """
        :param w: tuple d'observable
        :return: Matrice de dimension nb_d'etats * nb_d'etats * len(w) correspondant au xi du polycopié 4.4, sans boucle
        """
        f = self.f(w)[:, :-1]
        b = self.b(w)[:, 1:]
        emissions = self.emissions[:, w[1:]]
        xi = np.einsum('kt,kl,lt,lt->klt', f, self.transitions, emissions, b)
        v = np.einsum('kt,kl,lt,lt->t', f, self.transitions, emissions, b)
        somme = np.tile(v, (self.states_number, self.states_number, 1))
        xi = xi / somme
        return xi

    def xi2(self, w):
        """
        :param w: tuple d'observable
        :return: Matrice de dimension nb_d'etats * nb_d'etats * len(w) correspondant au xi du polycopié 4.4
        """
        f = self.f(w)[:, :-1]
        b = self.b(w)[:, 1:]
        emissions = self.emissions[:, w[1:]]
        xi = np.einsum('kt,kl,lt,lt->klt', f, self.transitions, emissions, b)
        for t in range(xi.shape[2]):
            xi[:, :, t] = xi[:, :, t] / np.sum(xi[:, :, t])
        return xi

    def bw1(self, S):
        """
        :param S: Liste de tuple d'observables
        :return: Modifie les valeurs des matrices du HMM pour augmenter la vraisemblance de S
        """
        if type(S) != list:
            raise TypeError("S doit être une liste")
        if len(S) == 0:
            raise ValueError("S ne doit pas être vide")
        for w in S:
            self.check_w(w)
        pi = np.zeros(self.states_number)
        T = np.zeros((self.states_number, self.states_number))
        O = np.zeros((self.states_number, self.letters_number))
        for j in range(len(S)):
            gamma = self.gamma(S[j])
            xi = self.xi(S[j])
            pi += np.array(gamma[:, 0])
            t = np.einsum('klt->kl', self.xi(S[j]))
            T += t
            for t in range(len(S[j])):
                O[:, S[j][t]] += gamma[:, t]
        self.transitions = (T.T / T.sum(1)).T
        self.emissions = (O.T / O.sum(1)).T
        self.initial = pi / pi.sum()

    @staticmethod
    def bw2(nbS, nbL, S, N):
        """
        :param nbS: Nombre d'états
        :param nbL: Nombre de sommets
        :param S: Liste de tuple d'observables
        :param N: nombre d iteration
        :return: Un HMM généré aléatoirement à nbS états et nbL sommets mis à jour N fois grâce à bw1 pour augmenter
        la vraisemblance
        """
        if type(N) != int or N < 0:
            raise ValueError("N doit être un entier positif")
        hmm = HMM.gen_HMM(nbL, nbS)
        for i in range(N):
            hmm.bw1(S)
        return hmm

    @staticmethod
    def bw3(nbS, nbL, S, N, M):
        """
        :param nbS: Nombre d'états
        :param nbL: Nombre de sommets
        :param S: Liste de tuple d'observables
        :param N: nombre d iterations
        :param M: nombre d initialisation differentes
        :return: Le HHMi avec 0 <= i <= M-1 qui maximise la vraisemblance de S
        """
        if type(M) != int or M < 0:
            raise ValueError("M doit être un entier positif")
        max_logV = 0
        hmm = None
        for i in range(M):
            print(i)
            h = HMM.bw2(nbS, nbL, S, N)
            logV = h.logV(S)
            if max_logV < logV:
                max_logV = logV
                hmm = h
        return hmm

    @staticmethod
    def bw2_variante(nbS, nbL, S, limite, N=None):
        # Même fonction que bw2 mais s'arrête automatiquement lorsque la log vraisemblance entre deux mises à jour
        # successives est inférieure au paramètre limite

        if N is not None and (type(N) != int or N < 0):
            raise ValueError("N doit être None ou un entier positif")
        if type(limite) != int and type(limite) != float or limite < 0:
            raise ValueError("le paramêtre limite doit être un nombre positif")

        hmm = HMM.gen_HMM(nbL, nbS)
        logV = hmm.logV(S)
        compteur = 0
        c = 0
        while compteur <= 10:
            if N is not None and c != N:
                break
            hmm.bw1(S)
            logV_new = hmm.logV(S)
            if logV - logV_new < limite:
                compteur += 1
            else:
                compteur = 0
            logV = logV_new
            if N is not None:
                c += 1
        return hmm

    @staticmethod
    def bw3_variante(nbS, nbL, S, M, limite, N=None):
        # Même fonction que bw3 mais s'arrête automatiquement lorsque la log vraisemblance entre deux mises à jour
        # successives est inférieure au paramètre limite
        if type(M) != int or M < 0:
            raise ValueError("M doit être un entier positif")
        max_logV = 0
        hmm = None
        for i in range(M):
            h = HMM.bw2_variante(nbS, nbL, S, limite, N)
            logV = hmm.logV(S)
            if max_logV < logV:
                max_logV = logV
                hmm = h
        return hmm

    @staticmethod
    def gen_HMM(nbr_lettre, nbr_etat):  # faire des checks sur les parametres
        """
        :param nbr_lettre: Nombre souhaité s'observables
        :param nbr_etat: Nombre souhaité d'états
        :return: un HMM généré aléatoirement avec nb_lettre observables et nb_ etats états
        """

        initial = HMM.list_rand_sum_2_dim(1, nbr_etat)
        transitions = HMM.list_rand_sum_2_dim(nbr_etat, nbr_etat)
        emissions = HMM.list_rand_sum_2_dim(nbr_etat, nbr_lettre)

        return HMM(nbr_lettre, nbr_etat, initial[0], transitions, emissions)

    def logV(self, S):
        """
        :param S: Liste de tuple d'observables
        :return: La log vraisemblance de S
        """
        somme = 0
        for w in S:
            self.check_w(w)
            somme += np.log(self.pfw(w))
        return somme

    @staticmethod
    def num_to_lettre(n):
        """
        :param n: Entier entre 0 et 25
        :return: La lettre de l'alphabet correspondant à n
        """
        if type(n) != int:
            raise TypeError('le numero doit etre un entier')
        if n < 0 or n > 25:
            raise ValueError('le numero doit etre compris entre 0 et 25')
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                    'u', 'v', 'w', 'x', 'y', 'z']
        return alphabet[n]

    @staticmethod
    def lettre_to_num(lettre):
        """
        :param lettre: Lettre de l'alphabet
        :return: L'indice de la lettre
        """
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                    'u', 'v', 'w', 'x', 'y', 'z']
        if type(lettre) != str:
            raise TypeError('la lettre doit etre un caractere')
        if lettre not in alphabet:
            raise ValueError('la lettre doit etre dans l\'alphabet')
        return alphabet.index(lettre)

    @staticmethod
    def list_rand_sum_2_dim(n, m):
        '''creer un array numpy en 2D aléatoire de dim n*m avec somme des valeurs = 1 sur toutes les lignes'''
        if type(n) != int:
            raise TypeError('n doit etre un entier')
        if type(m) != int:
            raise TypeError('m doit etre un entier')
        if n < 0:
            raise ValueError("n doit être positif")
        if m < 0:
            raise ValueError("m doit être positif")

        if m == 1:
            return np.array([[1] * n]) ####################### a verifier ###########################

        L = np.zeros((n, m - 1))

        for j in range(n):  # j colonne
            for i in range(m - 1):  # i sur la ligne
                s = random.random()
                s = "%.3f" % s
                L[j, i] = (s)

        for i in range(n):
            L.sort()

        M = np.zeros((n, m))

        for i in range(n):  # pour chaque ligne
            for j in range(0, m):  # pour chaque colonne
                if j == m - 1:
                    M[i, j] = 1 - L[i, j - 1]
                elif j == 0:
                    M[i, j] = L[i, j]
                else:
                    M[i, j] = L[i, j] - L[i, j - 1]

        return M
