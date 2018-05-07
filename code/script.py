###########################
# Fichier classe.py       #
# 06/05/18                #
# La communauté de l'info #
###########################

from classe import *

listeBW = []
input("Bonjour. Vous allez découvrir le code réalisé par la Communauté de l\'info pour le projet HMM. Pour continuer appuyez sur entrée.")
print()
adr = input("Veuillez entrer un chemin vers un HMM stocké sous format texte : ")
print()
h = HMM.load(adr)
print("Le HMM qui a été chargé est le suivant :")
print()
print(h)
print()
var = 'o'
while var == 'o':
    n = float(input("Veuillez entrer une longueur de mot à générer aléatoire grâce au HMM chargé : "))
    while n - int(n) != 0 or n <= 0:
        print ()
        print ("Vous devez entrer un entier positif")
        print ()
        n = float(input("Veuillez entrer une longueur de mot à générer aléatoire grâce au HMM chargé : "))
    n = int(n)
    print()
    w = h.generate_random(n)
    print("Le mot de", n, "lettre qui a été généré aléatoirement est", w)
    listeBW += [w]
    print("La probabilité que ce mot soit généré était de", h.pfw(w), ", soit une log-vraisemblance de", h.logV([w]))
    chemin, p = h.viterbi(w)
    print("Le cheminement d'états le plus probable pour la génération de ce mot est", chemin,
          "avec une probabilité logarithmique de", p)
    prochaine = h.predit(w)
    print("Si l'on continuait à générer des lettres pour ce mot, la prochaine serait le plus probalement", prochaine)
    print ()
    var = input("Voulez-vous générer une autre séquence afin d'utiliser Baum Welch ? o/n")
    while var != 'o' and var!= 'n':
        print ("Choisissez o ou n")
        print ()
        var = input("Voulez-vous générer une autre séquence afin d'utiliser Baum Welch ? o/n")
    print ()
nb = float(input("Combien de fois voulez-vous effectuer Baum Welch ?"))
while nb - int(nb) != 0 or nb < 0:
    print ()
    print ("Vous devez entrer un entier positif")
    print()
    nb = input ("Combien de fois voulez-vous effectuer Baum Welch ?")
nb = int(nb)
for i in range (nb):
    h.bw1(listeBW)
print()
print ("Le nouveau HMM dont la vraisemblance a été augmentée", nb, "fois est le suivant :")
print ()
print (h)


# bw, save