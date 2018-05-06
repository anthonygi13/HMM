###########################
# Fichier classe.py       #
# 06/05/18                #
# La communauté de l'info #
###########################

from classe import *

input("Bonjour. Vous allez découvrir le code réalisé par la Communauté de l\'info pour le projet HMM. Pour continuer appuyez sur entrée.")
print()
adr = input("Veuillez entrer un chemin vers un HMM stocké sous format texte : ")
print()
h = HMM.load(adr)
print("Le HMM qui a été chargé est le suivant :")
print()
print(h)
print()
n = int(input("Veuillez entrer une longueur de mot à générer aléatoire grâce au HMM chargé : "))
print()
w = h.generate_random(n)
print("Le mot de", n, "lettre qui a été généré aléatoirement est", w)
print("La probabilité que ce mot soit généré était de", h.pfw(w), ", soit une log-vraisemblance de", h.logV([w]))
chemin, p = h.viterbi(w)
print("Le cheminement d'états le plus probable pour la génération de ce mot est", chemin, "avec une probabilité logarithmique de", p)
prochaine = h.predit(w)
print("Si l'on continuait à générer des lettres pour ce mot, la prochaine serait le plus probalement", prochaine)





# bw, save