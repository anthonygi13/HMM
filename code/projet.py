
from classe import *
import matplotlib.pyplot as plt


def text_to_list(adr):
    data = open(adr, 'r')
    texte = data.read()
    L = texte.split('\n')
    data.close()
    return L


S = text_to_list('allemand2000')[:-1]
S2 = []
for w in S :
    w2 = ()
    for i in range (len(w)):
        w2 += (HMM.lettre_to_num(w[i]),)
    S2 += [w2]

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


