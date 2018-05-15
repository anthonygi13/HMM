from classe import *
import numpy as np
import matplotlib.pyplot as plt


def text_to_list(adr):
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
    return L2

def xval(nbFolds, S, nbL, nbSMin, nbSMax, nbIter, nbInit):
    n = len(S)
    l = np.random.permutation(n)
    lvOpt = -float("inf")
    for nbS in range(nbSMin, nbSMax):
        lv = 0
        for i in range(1,nbFolds+1):
            f1 = int((i-1)*n/nbFolds)
            f2 = int(i*n/nbFolds)
            learn = [S[l[j]] for j in range(f1)]
            learn += [S[l[j]] for j in range(f2,n)]
            test = [S[l[j]] for j in range(f1,f2)]
            h = HMM.bw3(nbL,nbS,learn,nbIter,nbInit)
        lv += h.logV(test)
        if lv > lvOpt:
            lvOpt = lv
            nbSOpt = nbS
    return lvOpt,nbSOpt

'''''''''''''''
y = []
x = []
for n in range (2,100000000):
    try:
        h = HMM.bw2_variante(n, 26, S2 , 1)
        y.append(h.logV(S2))
        x.append(n)
        print(y[-1])
    except KeyboardInterrupt:
        break

plt.plot(x, y)
plt.show()

'''''''''''''''''
