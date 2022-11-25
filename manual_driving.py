# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 16:02:24 2022

@author: aplissonneau
"""

from simulation.env import TrainEnv
import rllib_configs as configs
import keyboard
import time



policy_config = configs.APEX_TEST_CONFIG
policy_config["env_config"]["render"]["render"] = True

env = TrainEnv(policy_config["env_config"])

env.reset()

tx = time.time()

for i in range(1000):
    action = 1
    if keyboard.is_pressed('z'):
        action = 2
    
    if keyboard.is_pressed('e'):
        action = 0
        
    a= env.step(action)
    t2 = time.time()
    dt = t2 - tx
    time.sleep((i+1)/10 - dt)
    print('time', time.time() - tx)
    print(env.train.speed)


