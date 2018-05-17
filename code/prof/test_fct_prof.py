# Mac Donald  Kilian
# test_fct_prof
# crée le 16/05/2018
# def du fichier

# -*- coding: utf-8 -*-
"""
This module contains the HMM class
"""

import numpy as np

import random as rd

import math


class HMM:
    """ Define an automaton with parameters

    - Input:

    :param int nbL: the number of letters
    :param int nbS: the number of states
    :param vector initial: the initial vector
    :param array transitions: the transitions table
    :param array emissions: the emission tables

    """

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

    @property
    def nbL(self):
        return self.__nbL

    @nbL.setter
    def nbL(self, nbL_value):
        if (isinstance(nbL_value, int) and nbL_value > 0):
            self.__nbL = nbL_value
        else:
            raise ValueError("Error in new nbL value.")

    @property
    def nbS(self):
        return self.__nbS

    @nbS.setter
    def nbS(self, nbS_value):
        if (isinstance(nbS_value, int) and nbS_value >= 1):
            self.__nbS = nbS_value
        else:
            raise ValueError("Error in new nbS value.")

    @property
    def initial(self):
        return self.__initial

    @staticmethod
    def check_initial(values, nbS):
        if not isinstance(values, np.ndarray):
            raise TypeError("initial_values should be a numpy.array")
        if values.shape != (nbS,):
            mess = "The shape of initial_values should be "
            mess += str((nbS,))
            raise ValueError(mess)
        for i in range(nbS):
            if values[i] < 0:
                raise ValueError("Initial parameters should be positive")
        if not np.isclose(np.sum(values), 1.0):
            raise ValueError("Initial values should sum to 1")
        return values

    @initial.setter
    def initial(self, values):
        values = self.check_initial(values, self.nbS)
        self.__initial = values

    @property
    def transitions(self):
        return self.__transitions

    @staticmethod
    def check_transitions(values, nbL, nbS):
        if not isinstance(values, np.ndarray):
            raise TypeError("transitions_values should be a numpy.array")
        if values.shape != (nbS, nbS):
            mess = "The shape of transitions_values should be "
            mess += str((nbS, nbS))
            raise ValueError(mess)
        for i in range(nbS):
            for j in range(nbS):
                if values[i, j] < 0:
                    raise ValueError("Transition parameters \
                                      should be positive")
            if not np.isclose(np.sum(values[i, :]), 1.0):
                raise ValueError("Rows should sum to 1")
        return values

    @transitions.setter
    def transitions(self, values):
        values = self.check_transitions(values, self.nbL, self.nbS)
        self.__transitions = values

    @property
    def emissions(self):
        return self.__emissions

    @staticmethod
    def check_emissions(values, nbL, nbS):
        if not isinstance(values, np.ndarray):
            raise TypeError("emissions_values should be a numpy.array")
        if values.shape != (nbS, nbL):
            mess = "The shape of emissions_values should be "
            mess += str((nbS, nbL))
            raise ValueError(mess)
        for i in range(nbS):
            for j in range(nbL):
                if values[i, j] < 0:
                    raise ValueError("Transition parameters \
                                       should be positive")
            if not np.isclose(np.sum(values[i, :]), 1.0):
                raise ValueError("Rows should sum to 1")
        return values

    @emissions.setter
    def emissions(self, values):
        values = self.check_emissions(values, self.nbL, self.nbS)
        self.__emissions = values

    @staticmethod
    def load(adr):
        """ charger un HMM à partir d'un fichier """
        f = open(adr, "r")
        line = f.readline()
        while line[0] == "#":
            line = f.readline()
        line = line.split()
        nbL = int(line[0])
        line = f.readline()
        line = f.readline()
        line = line.split()
        nbS = int(line[0])
        initial = np.zeros([nbS])
        line = f.readline()
        for i in range(nbS):
            line = f.readline()
            line = line.split()
            initial[i] = float(line[0])
        transitions = np.zeros([nbS, nbS])
        line = f.readline()
        for i in range(nbS):
            line = f.readline()
            line = line.split()
            for j in range(nbS):
                transitions[i, j] = float(line[j])
        emissions = np.zeros([nbS, nbL])
        line = f.readline()
        for i in range(nbS):
            line = f.readline()
            line = line.split()
            for j in range(nbL):
                emissions[i, j] = float(line[j])
        f.close()
        return HMM(nbL, nbS, initial, transitions, emissions)

    def save(self, adr):
        " sauver un HMM dans un fichier """
        f = open(adr, "w")
        f.write("# The number of letters\n")
        f.write(str(self.nbL))
        f.write("\n# The number of states\n")
        f.write(str(self.nbS))
        f.write("\n# The initial transitions\n")
        for x in self.initial:
            f.write(str(x)+"\n")
        f.write("# The internal transitions\n")
        for i in range(self.nbS):
            for j in range(self.nbS):
                f.write(str(self.transitions[i, j])+" ")
            f.write("\n")
        f.write("# The emissions\n")
        for i in range(self.nbS):
            for x in self.emissions[i, :]:
                f.write(str(x)+" ")
            f.write("\n")
        f.close()

    @staticmethod
    def draw_multinomial(L):
        """ L est une liste de probabilités sommant à 1. Retourne un indice de la liste
        tiré aléatoirement selon la probabilité associée """
        x = rd.random()
        i = 0
        s = L[i]
        while x > s:
            i = i+1
            s = s + L[i]
        return i

    def gen_rand(self, n):
        """ retourne un mot de longueur n, tiré aléatoirement dans le HMM """
        s = self.draw_multinomial(self.initial)
        le = []
        w = []
        for i in range(n):
            le.append(s)
            w.append(self.draw_multinomial(self.emissions[s, :]))
            s = self.draw_multinomial(self.transitions[s, :])
        return tuple(le), tuple(w)

    def PFw(self, w):
        """ probabilité de w calculée selon la méthode forward """
        lp = self.initial.copy()
        lp = lp*self.emissions[:, w[0]]
        # lp[k] est la proba d'être dans l'état k
        # après avoir généré un préfixe de w
        for x in w[1:]:
            lp = np.dot(lp, self.transitions)*self.emissions[:, x]
        return lp.sum()

    def PBw(self, w):
        """ probabilité de w calculée selon la méthode backward """
        lp = self.emissions[:, w[-1]].copy()
        for x in reversed(w[:-1]):
            lp = np.dot(self.transitions, lp)
            lp = lp*self.emissions[:, x]
        return np.dot(self.initial, lp)

    def logV(self, S):
        """ log-vraisemblance d'un échantillon """
        l = 0
        for w in S:
            l += math.log(self.PFw(w))
        return l

    def predit(self, w):
        """ retourne la lettre la plus probable suivant le mot w """
        lp = self.initial.copy()
        for x in w:
            lp = lp*self.emissions[:, x]
            lp = np.dot(lp, self.transitions)
            lp = lp/sum(lp)
        return np.argmax(np.dot(lp, self.emissions))

    def Viterbi(self, w):
        lc = [[s] for s in range(self.nbS)]
        lp = [math.log(self.initial[s]) +
              math.log(self.emissions[s, w[0]])
              for s in range(self.nbS)]
        for x in w[1:]:
            newlc = []
            newlp = []
            for s in range(self.nbS):
                lps = [lp[sp] + math.log(self.transitions[sp, s]) +
                       math.log(self.emissions[s, x])
                       for sp in range(self.nbS)]
                pmax = max(lps)
                smax = lps.index(pmax)
                newlp.append(pmax)
                newlc.append(lc[smax]+[s])
            lp = newlp
            lc = newlc
        pmax = max(lp)
        smax = lp.index(pmax)
        return tuple(lc[smax]), pmax

    @staticmethod
    def succ(w, nbL):
        s = list(w)
        i = len(w) - 1
        while i >= 0 and s[i] == nbL-1:
            s[i] = 0
            i = i-1
        if i >= 0:
            s[i] += 1
        else:
            s.insert(0, 0)
        return tuple(s)

    @staticmethod
    def gen_vect(n):
        if n == 1:
            return np.ones(1)
        else:
            L = [(i, np.random.random()) for i in range(n-1)]
            L = sorted(L, key=lambda x: x[1])
            v = np.zeros(n)
            v[L[0][0]] = L[0][1]
            for i in range(1, n-1):
                v[L[i][0]] = L[i][1] - L[i-1][1]
            v[n-1] = 1 - L[n-2][1]
        return v

    @staticmethod
    def gen_HMM(nbL, nbS):
        initial = HMM.gen_vect(nbS)
        transitions = np.zeros((nbS, nbS))
        for i in range(nbS):
            transitions[i, :] = HMM.gen_vect(nbS)
        emissions = np.zeros((nbS, nbL))
        for i in range(nbS):
            emissions[i, :] = HMM.gen_vect(nbL)
        return HMM(nbL, nbS, initial, transitions, emissions)

    def forward(self, w):
        n = len(w)
        f = np.zeros([self.nbS, n])
        f[:, 0] = self.initial*self.emissions[:, w[0]]
        for i in range(1, n):
            f[:, i] = np.dot(f[:, i-1],
                             self.transitions)*self.emissions[:, w[i]]
        return f

    def backward(self, w):
        n = len(w)
        b = np.zeros([self.nbS, n])
        b[:, n-1] = np.ones(self.nbS)
        for i in range(1, n):
            b[:, n-i-1] = np.dot(self.transitions,
                                 b[:, n-i]*self.emissions[:, w[n-i]])
        return b

    def gamma(self, f, b):
        n = f.shape[1]
        v = np.array([np.dot(f[:, t], b[:, t]) for t in range(n)])
        gamma = (f*b)/v
        return gamma

    def xi(self, f, b, w):
        n = f.shape[1]
        xi = np.zeros([self.nbS, self.nbS, n-1])
        for t in range(n-1):
            for l in range(self.nbS):
                xi[:, l, t] = f[:, t]*self.transitions[:, l] * \
                              self.emissions[l, w[t+1]]*b[l, t+1]
            xi[:, :, t] /= xi[:, :, t].sum()
        return xi

    def BaumWelch_preparation(self, w):
        n = len(w)
        f = self.forward(w)
        b = self.backward(w)
        gamma = self.gamma(f, b)
        xi = self.xi(f, b, w)
        pinew = gamma[:, 0]
        Tnew = xi.sum(axis=2)
        W = np.zeros((n, self.nbL))
        for t in range(n):
                W[t, w[t]] = 1
        Enew = np.dot(gamma, W)
        return (pinew, Tnew, Enew)

    def BaumWelch(self, w):
        (pinew, Tnew, Enew) = self.BaumWelch_preparation(w)
        Tnew = (Tnew.T/Tnew.sum(axis=1)).T
        Enew = (Enew.T/Enew.sum(axis=1)).T
        return HMM(self.nbL, self.nbS, pinew, Tnew, Enew)

    def BaumWelchS(self, S):
        nbEx = len(S)
        pinew = np.zeros(self.nbS)
        Tnew = np.zeros((self.nbS, self.nbS))
        Enew = np.zeros((self.nbS, self.nbL))
        for w in S:
            (p, T, E) = self.BaumWelch_preparation(w)
            pinew += p
            Tnew += T
            Enew += E
        pinew /= nbEx
        Tnew = (Tnew.T/Tnew.sum(axis=1)).T
        Enew = (Enew.T/Enew.sum(axis=1)).T
        return HMM(self.nbL, self.nbS, pinew, Tnew, Enew)

    @staticmethod
    def BaumWelch2S(nbL, nbS, S, N):
        A = HMM.gen_HMM(nbL, nbS)
        for i in range(N):
            A = A.BaumWelchS(S)
        return A

    @staticmethod
    def BaumWelch3S(nbL, nbS, S, N, K):
        lv = -float('inf')
        for i in range(K):
            A = HMM.BaumWelch2S(nbL, nbS, S, N)
            lvnew = A.logV(S)
            if lvnew > lv:
                init = A.initial
                trans = A.transitions
                emis = A.emissions
                lv = lvnew
        return HMM(nbL, nbS, init, trans, emis)

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



def xval(nbFolds, S, nbL, nbSMin, nbSMax, nbIter, nbInit):
    n = len(S)
    l = np.random.permutation(n)
    lvOpt = -float('inf')
    for nbS in range(nbSMin, nbSMax):
        lv = 0
        for i in range(1, nbFolds + 1):
            f1 = int((i - 1) * n / nbFolds)
            f2 = int(i * n / nbFolds)
            learn = [S[l[j]] for j in range(f1)]
            learn += [S[l[j]] for j in range(f2, n)]
            test = [S[l[j]] for j in range(f1, f2)]
            h = HMM.BaumWelch3S(nbL, nbS, learn, nbIter, nbInit)
            lv += h.logV(test)
            if lv > lvOpt:
                lvOpt = lv
                nbSOpt = nbS
    return lvOpt, nbSOpt

def text_to_list(adr): # transforme un document texte contenant des mot en liste de mots compréhensibles par le HMM
    """
    :param adr: addresse du fichier texte à convertir
    :return: liste de tuples correspondant aux mots se trouvant dans le fichier texte
    """
    data = open(adr, 'r')
    texte = data.read()
    L = texte.split('\n')
    data.close()
    L2 = []
    for w in L:
        w2 = ()
        for i in range (len(w)):
            w2 += (HMM.lettre_to_num(w[i]),)
        L2 += [w2]
    return L2[:-1]















L = text_to_list('anglais2000')
print('toc',xval(20, L, 26, 2, 10, 5, 10))