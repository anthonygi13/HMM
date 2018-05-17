###########################
# Fichier projet.py       #
# 16/05/18                #
# La communauté de l'info #
###########################

from classe import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import random


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


def xval(nbFolds, S, nbL, nbSMin, nbSMax, nbIter, nbInit):
    n = len(S)
    l = np.random.permutation(n)
    lvOpt = -float("inf")
    for nbS in range(nbSMin, nbSMax + 1):
        print(nbS)
        lv = 0
        for i in range(1,nbFolds+1):
            f1 = int((i-1)*n/nbFolds)
            f2 = int(i*n/nbFolds)
            learn = [S[l[j]] for j in range(f1)]
            learn += [S[l[j]] for j in range(f2,n)]
            test = [S[l[j]] for j in range(f1,f2)]
            h = HMM.bw3(nbS, nbL, learn, nbIter, nbInit)
            lv += h.logV(test)
        if lv > lvOpt:
            lvOpt = lv
            nbSOpt = nbS
    return lvOpt, nbSOpt


def func(x, a, b, c): #fonction utilisée pour le fitting de certaines courbes
    return - a * np.exp(-x / b) - 25580 + c

def logV_vs_nb_iteration_bw1(nb_iter_max, nbS, S, nbL=26): # trace la log vraisemblance en fonction du nombre d'itération de bw1
    """
    :param nb_iter_max: nombre d'itérations de bw1 à réaliser
    :param nbS: nb d'états
    :param S: liste de mots sur laquelle on entraine notre HMM
    :param nbL: nombre de lettres
    """

    hmm = HMM.gen_HMM(nbL, nbS)
    nb_iter = [0]
    logV = [hmm.logV(S)]
    for i in range(1, nb_iter_max + 1):
        print(i)
        try:
            hmm.bw1(S)
            nb_iter.append(i)
            logV.append(hmm.logV(S))
        except KeyboardInterrupt:
            break
    plt.plot(nb_iter, logV, '.', c='blue', label='logV en fonction du nombre d\'itération de bw1')
    plt.xlabel('nb d\'iteration')
    plt.ylabel('logV')
    titre = 'anglais2000' + ' / nombre d\'etat = ' + str(nbS)
    plt.title(titre)
    optimizedParameters, pcov = opt.curve_fit(func, nb_iter, logV)
    print(optimizedParameters)

    print(nb_iter)
    print(logV)

    # Use the optimized parameters to plot the best fit
    plt.plot(nb_iter, [func(x, optimizedParameters[0], optimizedParameters[1], optimizedParameters[2]) for x in nb_iter], label='-' + str(optimizedParameters[0]) + 'exp(-x/' + str(optimizedParameters[1]) + ') + ' + str(-25580 + optimizedParameters[2]))

    plt.legend()
    plt.show()


def logV_vs_intialisation(nb_init_max, nb_iter, nbS, S, nbL=26): # trace la logvraisemblance optimale en fonction de différentes initialisations
    """
    :param nb_init_max: nombre d'initialisations différentes à réaliser
    :param nb_iter: nombre d'itération dans bw2
    :param nbS: nombre d'états
    :param S: liste de mots sur laquelle on entraine nos HMM
    :param nbL: nombre de lettres
    """

    nb_init = []
    logV = []
    for i in range(1, nb_init_max + 1):
        try:
            h = HMM.bw2(nbS, nbL, S, nb_iter)
            nb_init.append(i)
            logV.append(h.logV(S))
        except KeyboardInterrupt:
            break
    plt.plot(nb_init, logV)
    plt.show()


def logV_vs_initialisation_variante(nb_init_max, limite, nbS, S, nbL=26): # trace la logvraisemblance optimale en fonction de différentes initialisations
    """
    :param nb_init_max: nombre d'initialisations différentes à réaliser
    :param limite: limite pour bw2_variante
    :param nbS: nombre d'états
    :param S: liste de mots sur laquelle on entraine nos HMM
    :param nbL: nombre de lettres
    """

    nb_init = []
    logV = []
    for i in range(1, nb_init_max + 1):
        try:
            h = HMM.bw2_variante(nbS, nbL, S, limite)
            nb_init.append(i)
            logV.append(h.logV(S))
        except KeyboardInterrupt:
            break
    plt.plot(nb_init, logV)
    plt.show()


def efficiency_vs_nb_state(nbFolds, S, nbSMin, nbSMax, nbIter, nbInit, nbL=26):  # trace la log vraisemblance moyenne sur les echantillons tests en fonction du nombre d'état
    """
    :param nbFolds: cardinal de la partition de S
    :param S: liste de mots sur laquelle on entraine notre HMM
    :param nbSMin: nombre d'etat minimum
    :param nbSMax: nombre d'etat maximum
    :param nbIter: nombre d'itérations pour bw3
    :param nbInit: nbombre d'initialisations pour bw3
    :param nbL: nombre de lettres pour le HMM
    """

    n = len(S)
    l = np.random.permutation(n)
    nb_state = []
    logV = []
    for nbS in range(nbSMin, nbSMax + 1):
        print(nbS)
        try:
            lv = 0
            for i in range(1, nbFolds + 1):
                f1 = int((i - 1) * n / nbFolds)
                f2 = int(i * n / nbFolds)
                learn = [S[l[j]] for j in range(f1)]
                learn += [S[l[j]] for j in range(f2, n)]
                test = [S[l[j]] for j in range(f1, f2)]
                h = HMM.bw3(nbS, nbL, learn, nbIter, nbInit)
                lv += h.logV(test)
            logV.append(lv / nbFolds)
            print("logV", lv/nbFolds)
            nb_state.append(nbS)
        except KeyboardInterrupt:
            break
    plt.plot(nb_state, logV)
    plt.show()


def efficiency_vs_nb_state_variante(nbFolds, S, nbSMin, nbSMax, limite, nbInit,
                           nbL=26):  # trace la log vraisemblance moyenne sur les echantillons tests en fonction du nombre d'état
    """
    :param nbFolds: cardinal de la partition de S
    :param S: liste de mots sur laquelle on entraine notre HMM
    :param nbSMin: nombre d'etat minimum
    :param nbSMax: nombre d'etat maximum
    :param limite: limite pour bw3_variante
    :param nbInit: nbombre d'initialisations pour bw3
    :param nbL: nombre de lettres pour le HMM
    """

    n = len(S)
    l = np.random.permutation(n)
    nb_state = []
    logV = []
    for nbS in range(nbSMin, nbSMax + 1):
        print(nbS)
        try:
            lv = 0
            for i in range(1, nbFolds + 1):
                f1 = int((i - 1) * n / nbFolds)
                f2 = int(i * n / nbFolds)
                learn = [S[l[j]] for j in range(f1)]
                learn += [S[l[j]] for j in range(f2, n)]
                test = [S[l[j]] for j in range(f1, f2)]
                h = HMM.bw3_variante(nbS, nbL, learn, nbInit, limite)
                lv += h.logV(test)
            logV.append(lv / nbFolds)
            print("logV", lv / nbFolds)
            nb_state.append(nbS)
        except KeyboardInterrupt:
            break
    plt.plot(nb_state, logV)
    plt.show()

#L = text_to_list('anglais2000')
#print('toc', xval(20, L, 26, 2, 10, 5, 5))

#logV_vs_nb_iteration_bw1(1000, 30, text_to_list('anglais2000'))


#efficiency_vs_nb_state(10, text_to_list('anglais2000'), 53, 1000, 100, 1)

#efficiency_vs_nb_state(10, text_to_list('allemand2000'), 2, 1000, 100, 1)


#logV_vs_nb_iteration_bw1(1000, 45, text_to_list('anglais2000'))



#HMM.bw3(45, 26, text_to_list('anglais2000'), 200, 20).save("hmm_anglais_v2")


"""
h = HMM.load("hmm_anglais")

print(h.logV(text_to_list('anglais2000')))

for i in range(1000000):
    n = random.randint(3, 8)
    print(h.gen_mot_lettres(n))
"""