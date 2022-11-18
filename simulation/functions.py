# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 18:04:29 2020

@author: aplissonneau
"""
import pickle
import numpy as np
import os
import pandas as pd



path = os.path.dirname(__file__)
print(path)
#model = pickle.load(open(os.path.join(path,'freinage'),'rb')) #modèle de freinage

#import abaque for traction_TFA and freinage_TFA

traction_csv = os.path.join(path, "traction.csv")
freinage_csv = os.path.join(path, "dynamic_braking.csv")
freinage_ind_csv = os.path.join(path, "indirect_braking.csv")

abaque_traction = pd.read_csv(traction_csv)
abaque_traction = abaque_traction.iloc[-1,2:]

abaque_freinage = pd.read_csv(freinage_csv)
abaque_freinage = abaque_freinage.iloc[-1,2:]

abaque_freinage_ind = pd.read_csv(freinage_ind_csv)
abaque_freinage_ind = abaque_freinage_ind.iloc[1,2:]


A = 0.0912
B = 0.01
C= 0.0001793

M_loco = 90e3
m_wagon = 50e3
n_wagon = 20
M = M_loco + m_wagon * n_wagon

def traction_TFA(Vkmhr):
    assert Vkmhr <= 100, f"Speed lower than 100km/h expected, got {Vkmhr}"
    assert Vkmhr >= 0, f"Speed higher than 0 km/h expected, got {Vkmhr}"
    V = Vkmhr/3.6
    F_res = A + B * V + C * V**2 
    F_trac = abaque_traction.iloc[int(Vkmhr//5)] * 1000
    F = F_trac - F_res
    acc = F / M
    return acc    


def freinage_TFA(Vkmhr):
    assert Vkmhr <= 100, f"Speed lower than 100km/h expected, got {Vkmhr}"
    assert Vkmhr >= 0, f"Speed higher than 0 km/h expected, got {Vkmhr}"
    V = Vkmhr/3.6
    F_res = A + B * V + C * V**2 
    F_frei = abaque_freinage.iloc[int(Vkmhr//5)] * 1000
    F_frei_ind = abaque_freinage_ind.iloc[int(Vkmhr//5)] * 1000 * n_wagon
    F = - F_frei - F_frei_ind - F_res
    acc = F / M
    return acc    


K = 0.07
n = 6
W = 196
w = W / n
noCars = 50 #Nombre de wagon
wPerCar = 130 #Poids de chaque wagon

P = 6000 # horsepower (hp)
nu = 0.88 # efficiency

p_m_f = 0.97
regime_p = 0

def traction(Vkmhr):
    #valeur realiste d'une loco en particulier.
    if Vkmhr < 19:
        T = (2650 * nu * P) / 19  # in N
    else:
        T = (2650 * nu * P) / Vkmhr  # in N
    V = Vkmhr / 3.6
    R_p = 0.6 + (20 / w) + (0.01 * V) + (K * V**2) / (w * n)
    R = R_p * (50 * 130 + 196) * 4.4 # en N
    a = (T - R ) / ((noCars * wPerCar + W) * 1000)
    a = a * 3.6
    a = a if a > 0 else 0
    return a





def traction_tram(Vkmhr):
    if Vkmhr < 40:
        return 0.92 #* 3.6
    elif Vkmhr > 40 and Vkmhr < 70:
        return 0.67 #* 3.6
    else:
        return 0

def freinage_tram(Vkmhr):
    return -1.50 #*3.6

def freinage_rgv(Vkmhr):
    return -0.5

def traction_rgv(Vkmhr):
    return 0.5

#def freinage(V):
#    #valeur de loco.
#    V=V+1
#    X = np.array([regime_p,V,noCars,p_m_f]).reshape(1,-1)
#    X = model.poly.transform(X)
#    X = model.scaler.transform(X)
#    y = model.predict(X)
#
#    a = -(V)**2/(2*y[0])
#    return a


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    V = [x for x in range(200)]
#    a = [traction(x) for x in V]
#    plt.plot(V,a)
#    V=0
#    for i in range(100):
#        print(V)
#        V += traction(V)
#        time.sleep(1)
    a = [freinage(x) for x in V]
    plt.plot(V,a)
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Acceleration (m.s-2)")


