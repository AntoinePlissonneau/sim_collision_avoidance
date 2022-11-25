# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 17:07:00 2022

@author: aplissonneau
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper') 
import seaborn as sns
import numpy as np

params = {'legend.fontsize': '20',
         'axes.labelsize': '35',
         'figure.figsize': (14, 10),
         'axes.titlesize':'1',
         'xtick.labelsize':'25',
         'ytick.labelsize':'25'}
plt.rcParams.update(params)

data = pd.read_csv("data/collision_runs.csv").iloc[:,1:]



b = (data.iloc[-1,:] - data.iloc[-50,:]).dropna()/8
c = (data.iloc[-1,:] - data.iloc[-20,:]).dropna()/8
d = (data.iloc[-1,:] - data.iloc[-10,:]).dropna()/8
plt.figure()
plt.boxplot([b,c,d], labels=["-5s","-2s","-1s"])
plt.xlabel("Seconds before collision reference", labelpad=10)
plt.ylabel("Speed reduction", labelpad=10)
#plt.title("Boxplots of speed reduction before collision")
plt.show()

col_speed = data.iloc[-1,:].dropna()




plt.figure()

sns.kdeplot(col_speed, label="Collision speed distribution")
plt.vlines(2.16,-0.1,1,"red","dashed", label="Mean speed CNN-LSTM with aux task")

plt.xlabel("Speed (m/s)", labelpad=10)
plt.ylabel("Density", labelpad=10)
plt.legend()
plt.show()






a = np.array([(data.iloc[-i-1,:].dropna()).mean() for i in range(50)])
b = np.array([(data.iloc[-i-1,:].dropna()).std() for i in range(50)])

plt.figure()
#plt.boxplot(col_speed)
#plt.title("Speed curve before collision")
plt.plot([0.1*i - 5 for i in range(len(a))],np.flip(a))


#plt.hlines(0.27,-1,8,"red","dashed", label="Mean speed auxCNN-LSTM")
plt.ylabel("Speed (m/s)", labelpad=10)
plt.xlabel("Time before collision (s)", labelpad=10)
#plt.ylim([0,7.5])
plt.show()





plt.figure()
plt.plot(data.iloc[-10,:], data.iloc[-1,:], "o")
plt.xlim([0,5])
plt.ylim([0,5])
plt.plot([0.0, 5.0], [0.0, 5.0])
plt.xlabel(r"$v(t_{col})$ (m/s)", labelpad=10)
plt.ylabel(r"$v(t_{col}-1s)$ (m/s)", labelpad=10)
plt.show()

plt.figure()
plt.plot(data.iloc[-20,:], data.iloc[-10,:], "o")
plt.xlim([0,5])
plt.ylim([0,5])
plt.plot([0.0, 5.0], [0.0, 5.0])
plt.xlabel(r"$v(t_{col}-1s)$ (m/s)", labelpad=10)
plt.ylabel(r"$v(t_{col}-2s)$ (m/s)", labelpad=10)
plt.show()

plt.figure()
plt.plot(data.iloc[-30,:], data.iloc[-20,:], "o")
plt.xlim([0,5])
plt.ylim([0,5])
plt.plot([0.0, 5.0], [0.0, 5.0])
plt.xlabel(r"$v(t_{col}-2s)$ (m/s)", labelpad=10)
plt.ylabel(r"$v(t_{col}-3s)$ (m/s)", labelpad=10)
plt.show()


plt.figure()
plt.plot(data.iloc[-40,:], data.iloc[-30,:], "o")
plt.xlim([0,5])
plt.ylim([0,5])
plt.plot([0.0, 5.0], [0.0, 5.0])
plt.xlabel(r"$v(t_{col}-3s)$ (m/s)", labelpad=10)
plt.ylabel(r"$v(t_{col}-4s)$ (m/s)", labelpad=10)
plt.show()






