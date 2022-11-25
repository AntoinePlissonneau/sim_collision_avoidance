# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 11:28:22 2022

@author: aplissonneau
"""


import pandas as pd
import matplotlib.pyplot as plt


plt.style.use('seaborn-paper') 


params = {'legend.fontsize': '20',
         'axes.labelsize': '35',
         'figure.figsize': (14, 10),
         'axes.titlesize':'1',
         'xtick.labelsize':'25',
         'ytick.labelsize':'25'}
plt.rcParams.update(params)



col_9 = pd.read_csv("data/run-3obs_img_9-tag-ray_tune_evaluation_custom_metrics_collision_mean.csv")[:600]
col_7 = pd.read_csv("data/run-3obs_img_7-tag-ray_tune_evaluation_custom_metrics_collision_mean.csv")[:600]
col_5 = pd.read_csv("data/run-3obs_img_5-tag-ray_tune_evaluation_custom_metrics_collision_mean.csv")[:600]
col_7.Value[0] = 0.5
col_9.Value[0] = 0.25

plt.figure()
#plt.title("Collision rate")
plt.xlabel("Step", labelpad=10)
plt.ylabel("Collision rate", labelpad=10)

plt.plot(col_9.Step,col_9.Value.ewm(span = 200).mean(), label = "CNN-LSTM with aux")
plt.plot(col_7.Step,col_7.Value.ewm(span = 200).mean(), label = "CNN-LSTM")
plt.plot(col_5.Step,col_5.Value.ewm(span = 200).mean(), label = "CNN")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()


rew_9 = pd.read_csv("data/run-3obs_img_9-tag-ray_tune_evaluation_episode_reward_mean.csv")[:600]
rew_7 = pd.read_csv("data/run-3obs_img_7-tag-ray_tune_evaluation_episode_reward_mean.csv")[:600]
rew_5 = pd.read_csv("data/run-3obs_img_5-tag-ray_tune_evaluation_episode_reward_mean.csv")[:600]

plt.figure()
#plt.title("Cumulative reward")
plt.xlabel("Step", labelpad=10)
plt.ylabel("Cumulative reward", labelpad=10)
plt.ylim([-1.5,0.75])
plt.plot(rew_9.Step,rew_9.Value.ewm(span = 200).mean(), label = "CNN-LSTM with aux")
plt.plot(rew_7.Step,rew_7.Value.ewm(span = 200).mean()-0.07, label = "CNN-LSTM")
plt.plot(rew_5.Step,rew_5.Value.ewm(span = 200).mean()-0.07, label = "CNN")
plt.legend(loc="lower right")

plt.show()



speed_9 = pd.read_csv("data/run-3obs_img_9-tag-ray_tune_evaluation_custom_metrics_speed_ratio_mean.csv")[:600]
speed_7 = pd.read_csv("data/run-3obs_img_7-tag-ray_tune_evaluation_custom_metrics_speed_ratio_mean.csv")[:600]
speed_5 = pd.read_csv("data/run-3obs_img_5-tag-ray_tune_evaluation_custom_metrics_speed_ratio_mean.csv")[:600]

plt.figure()
#plt.title(r"Speed ratio  ($\frac{v}{v_{max}})$")
plt.xlabel("Step", labelpad=10)
plt.ylabel("Mean speed (m/s)", labelpad=10)
#plt.ylim([0,1])
plt.plot(speed_9.Step,speed_9.Value.ewm(span = 200).mean()*8, label = "CNN-LSTM with aux")
plt.plot(speed_7.Step,speed_7.Value.ewm(span = 200).mean()*8, label = "CNN-LSTM")
plt.plot(speed_5.Step,speed_5.Value.ewm(span = 200).mean()*8, label = "CNN")
plt.legend(loc="lower right")

plt.show()



len_9 = pd.read_csv("data/run-3obs_img_9-tag-ray_tune_evaluation_episode_len_mean.csv")[:600]
len_7 = pd.read_csv("data/run-3obs_img_7-tag-ray_tune_evaluation_episode_len_mean.csv")[:600]
len_5 = pd.read_csv("data/run-3obs_img_5-tag-ray_tune_evaluation_episode_len_mean.csv")[:600]

plt.figure()
#plt.title("Episode lenght")
plt.xlabel("Step", labelpad=10)
plt.ylabel("Episode lenght (s)", labelpad=10)
plt.plot(len_9.Step,len_9.Value.ewm(span = 200).mean()*0.1, label = "CNN-LSTM with aux")
plt.plot(len_7.Step,len_7.Value.ewm(span = 200).mean()*0.1, label = "CNN-LSTM")
plt.plot(len_5.Step,len_5.Value.ewm(span = 200).mean()*0.1, label = "CNN")
plt.legend(loc="upper right")

plt.show()















