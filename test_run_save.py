# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 19:11:14 2022

@author: aplissonneau
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:51:37 2021

@author: aplissonneau
"""
REFRESH_TIME = 0.1
import gym
from gym import spaces
import numpy as np
import random
import copy
import glob
import re



    
def visualize_agent(env_test, model, load_path = False, env_id = 'myenv-v0'):
    obs = env_test.reset()
    done = False
    rew_cum = 0
    i2 = 0
    for i in range(2000):
#        img = np.transpose(obs["img"], (1,2,0))
#        width = int(img.shape[1] * 5)
#        height = int(img.shape[0] * 5)
#        dim = (width, height)
##          
#        # resize image
#        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#        cv2.imshow("ok", resized)
#        cv2.waitKey(50)
#
        action = model.compute_action(obs)
#        print(action)
        obs, rewards, done, info = env_test.step(action)
        plt.savefig(f"img/video/{i}.png")
        #plt.show()
        rew_cum += info["reward_speed"]
        i2+=1
        if i2 > 1250:
            done = True
        if i % 100 == 0:
            print(rew_cum)
        if done:
#            plt.clf()
            i2=0
            rew_cum = 0
            env_test.reset()

    numbers = re.compile(r'(\d+)')
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
    
    img_array = []
    for filename in sorted(glob.glob('img/video/*.png'), key=numericalSort):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
 
 
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def eval_agent(env_test, model, ep=5):
    
    hist = {"collision":[],"timeout":[], "speed_ratio":[], "col_speed":[]}
    
    speed_d = []
    for i in range(ep):
        speed = []
        obs = env_test.reset()
        done = False
        speed_ratio = []
        if i%50==0:
            print(i)
        while not done:
            
            action = model.compute_action(obs)
            obs, rewards, done, info = env_test.step(action)
            
            speed.append(info["speed_ratio"])
            
            if done:
                hist["collision"].append(info["collision"])
                hist["col_speed"].append(info["col_speed"])
                hist["timeout"].append(info["timeout"])   
                hist["speed_ratio"].append(np.mean(speed_ratio))
                speed_d.append(speed)

    return hist, speed_d
                
def eval_ini(ini_config):
    dico = {}
    for key in ini_config:
        d = {}
        for k in ini_config[key]:
            try:
                d[k] = eval(ini_config[key][k])
            except:
                d[k] = ini_config[key][k]
        dico[key] = d
    return dico

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    import configparser
    from observation_builder.obs_builder import ImgObservationBuilder, SimpleObservationBuilder, ImgSeqObservationBuilder, ImgStackObservationBuilder
    from simulation.env import TrainEnv
    from custom_policies_ray import LowresCNN
    import cv2
    #    multiprocessing.freeze_support()
    import ray.rllib.agents.dqn as dqn
    import ray
    from ray import tune
    from ray.rllib.models import ModelCatalog
    import json
    import pandas as pd
    
    config = configparser.ConfigParser()
    config.read("/home/aplissonneau/simulateur/params_RL.ini")
    config  = eval_ini(config)
#    config = {}
    config_ray_file = "/home/aplissonneau/simulateur/rllib_test/3obs_img/9/params.json"
    f=open(config_ray_file)
    config_ray = json.load(f)
    
    
    policy_config = dqn.APEX_DEFAULT_CONFIG
    policy_config["num_workers"] = 1
    policy_config["buffer_size"] = 100000
    
    policy_config['train_batch_size']= 32
    
    policy_config["env_config"] = config
    policy_config["env_config"]["env"]["obs_builder"] = ImgSeqObservationBuilder()
    policy_config["framework"] = "torch"
    #    policy_config["run"]=dqn.ApexTrainer
    policy_config["env"]=TrainEnv
    policy_config["horizon"]=1200
    policy_config["batch_mode"]= "complete_episodes"
    
    policy_config["evaluation_interval"]= 2
    policy_config["evaluation_duration"]= 30
    
    policy_config["evaluation_duration_unit"]= "episodes"
    policy_config['evaluation_config']= {'explore': False}
    policy_config['evaluation_num_workers']= 1
    #    policy_config['rollout_fragment_length']= 200
    #    policy_config["model"]["custom_model"] = "LowresCNNLSTM"
    policy_config['log_level']= 'ERROR'
    
    policy_config["model"]["custom_model"] = config_ray["model"]["custom_model"]
    
    test_agent = dqn.ApexTrainer(env=TrainEnv, config=policy_config)
    test_agent.restore("/home/aplissonneau/simulateur/rllib_test/3obs_img/9/checkpoint_003600/checkpoint-3600")
    policy_config["env_config"]["render"]["render"] = True
    env_test = TrainEnv(policy_config["env_config"])
    
    visualize = False
    if visualize:
        visualize_agent(env_test, test_agent)
    else:
        
        policy_config["env_config"]["render"]["render"] = False
        env_test = TrainEnv(policy_config["env_config"])
    
        hist, speed = eval_agent(env_test, test_agent, ep=30)
        print(speed)
        pd.DataFrame(speed).T.to_csv("analysis/runs9_3obs.csv")
        #print(hist)
        print("collision:",np.mean(hist["collision"]))
        print("timeout:",np.mean(hist["timeout"]))
        print("speed_ratio:",np.mean(hist["speed_ratio"]))
        col_speed =  [i for i in hist["col_speed"] if i is not None]
        print(np.mean(col_speed))
        
        
        