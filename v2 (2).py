import random
from scipy import integrate
from math import sqrt, pi, exp
import matplotlib.pyplot as plt
import numpy as np



#Un des paramètres que nous pouvons fixer est le point d'indice qui vaut pour l'instant:
point_d_indice=4.85003

def gaussienne(x):
    # densité de probabilité de la loi normale qui sera utiliser pour distribuer les ages.
    return exp(-(x-40)**2/(2*400))/(sqrt(2*pi)*20)

def age_random(f):

    y = random.random()
    x = 20
    w = integrate.quad(f, 0, x)

    while abs(w[0]-y) > 0.1:
        # on cherche la valeur de x pour laquelle P( X >= x ) = y
        x += 1/12
        w = integrate.quad(f, 0, x)
        if x <= 20:
            return 20
        if x >= 64:
            return 64
    return x

membres_du_conseil_détat = [219,219,219,219,219,219,219,219,219,219,222,219,222,227,229,229,228,234]

magistrats_de_lordre_administratif = [964,964,990,1019,1058,1083,1108,1126,1146,1156,1176,1203,1229,1238,1246,1253,1255,1271]

catégorie_A = [300, 305, 294, 327, 472, 524, 751, 817, 832, 835, 845, 844, 880, 896, 984, 1037, 1058, 1060]

catégorie_B = [307, 311, 299, 333, 357, 363, 363, 358, 367, 371, 378, 385, 391, 409, 433, 443, 446, 452]

catégorie_C = [1031, 1037, 1050, 1060, 1184, 1154, 1137, 1135, 1149, 1157, 1163, 1168, 1177, 1183, 1233, 1262, 1266, 1269]

point_dindice = [5181.75, 5212.84, 5249.33, 5275.58, 5301.96, 5328.47, 5371.1, 5397.95, 5441.13, 5468.34, 5484.75, 5512.17, 5528.71, 5556.35, 5589.69, 5623.23, 5820.04]

def moyenne(L):
    S = 0
    for k in L:
        S += k
    return(S/len(L))

def ecart_type(L):
    S = 0
    M = moyenne(L)
    for k in L:
        S += (k - M)**2
    return sqrt(S)
#
P = [5181.75]
for k in range(49):
    P.append(P[-1]+np.random.normal(50, 670/8))

plt.plot(range(50), P)
plt.show()







Années = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

coefficients = np.polyfit(Années, catégorie_A, deg=2)
[a1, a2, a3] = coefficients


y_pred = [a1*k**2 + a2*k**1 + a3 for k in np.linspace(2006,2030, 100)]

plt.scatter(Années, catégorie_A, s=10)
plt.plot(np.linspace(2006,2030, 100), y_pred, color='r')
plt.show()

