###########################
# Fichier classe.py       #
# 06/05/18                #
# La communauté de l'info #
###########################

from classe import *

def entree():
    while True:
        try:
            x = int(input(""))
            if x <= 0:
                raise ValueError
            break
        except ValueError:
            print("Oups ! Vous devez entrer un nombre entier strictement positif, essayez encore...")
    return x

def verif_entree(entree, texte):
    while entree - int(entree) != 0 or entree <= 0:
        print ()
        print ("Vous devez entrer un entier strictement positif")
        print ()
        entree = float(input(texte))
    return entree

listeBW = []
input("Bonjour. Vous allez découvrir le code réalisé par la Communauté de l\'info pour le projet HMM. Pour continuer appuyez sur entrée.")
print()
while True:
    try:
        adr = input("Veuillez entrer un chemin vers un HMM stocké sous format texte : ")
        h = HMM.load(adr)
        break
    except:
        print("Le chemin n'est pas valide, essayez encore...")

print()
print("Le HMM qui a été chargé est le suivant :")
print()
print(h)

var = 'o'
while var == 'o':
    print()
    print("Veuillez entrer une longueur de mot à générer aléatoire grâce au HMM chargé : ")
    n = entree()
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
    var = input("Voulez-vous générer une autre séquence afin d'utiliser Baum Welch (o/n) ? ")

    while var != 'o' and var!= 'n':
        print ("Choisissez o ou n...")
        print ()
        var = input("Voulez-vous générer une autre séquence afin d'utiliser Baum Welch (o/n) ? ")

print()
print("La log-vraisemblance de l'échantillon", listeBW, "est de", h.logV(listeBW))
print("Combien de fois voulez-vous effectuer Baum Welch ? ")
nb = entree()

print()
for i in range (nb):
    h.bw1(listeBW)
    print("étape", i+1, ": La log-vraisemblance de l'échantillon", listeBW, "est de", h.logV(listeBW))

print()
print ("Le nouveau HMM pour lequel la vraisemblance de l'échantillon", listeBW, "a été augmentée", nb, "fois est le suivant :")
print ()
print (h)
print()
print("La vraisemblance de l'échantillon", listeBW, "est désormais de", h.logV(listeBW))

var = input("Voulez vous sauvergarder cet HMM (o/n)?")
if var =='o':
    name=input('Entrez le nom du fichier : ')
    h.save(name)
else:
    print("Merci d'avoir utilisé notre programme !")

# bw, save