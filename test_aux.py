import gym
from gym import spaces
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import os
import configparser
from observation_builder.obs_builder import ImgObservationBuilder, SimpleObservationBuilder, ImgSeqObservationBuilder
from simulation.env import TrainEnv
from custom_policies_ray import LowresCNN, LowresCNNLSTM, LowresCNNLSTM_test
import cv2
#    multiprocessing.freeze_support()
import ray.rllib.agents.dqn as dqn
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
import torch




def img_mask(train_speed,
             prev_obs,
             env_rate=10,
             future_step=4,
             obs_max_speed = 3):
    def gla(train_speed, step, acc, env_rate):
        speed = train_speed
        dist = 0
        for i in range(step):

            dist += speed/10
            speed = max(0, min(speed+acc/10, 7))

        return dist
    curr_img = np.transpose(prev_obs[0][3], (1,2,0))
    img = curr_img[:,:,0]
    coords_obs = np.argwhere(prev_obs2[:,:,0] > 0)
    train_max_dist = gla(train_speed, future_step, 0.9, env_rate)
    train_min_dist = gla(train_speed, future_step, -1.5, env_rate)

    max_coords_x = coords_obs[:,1] - train_min_dist
    min_coords_x = coords_obs[:,1] - train_max_dist
    r = future_step/env_rate * obs_max_speed
    max_coords_y = coords_obs[:,0] + r
    min_coords_y = coords_obs[:,0] - r

    max_coords = coords_obs - (r, train_min_dist)
    min_coords = coords_obs - (- r, train_max_dist)

    in_square = lambda x: x > min_coords_x and x < max_coords_x and y > min_coords_y and y < max_coords_y

    max_circle_center = min_circle_center = coords_obs.copy() 
    max_circle_center[:,1] = max_coords[:,1]
    min_circle_center[:,1] = min_coords[:,1]
    
    max_dist = future_step/env_rate * obs_max_speed

    centers = coords_obs
    r = max_dist
    s = img.shape
    coords = []
    for i in range(len(centers)):
        for y in range(s[0]):
            for x in range(s[1]):
                in_square = x > min_coords[i][1] and x < max_coords[i][1] and y > min_coords[i][0] and y < max_coords[i][0]
                in_max_circle = (y-max_circle_center[i][0]) **2 + (x-max_circle_center[i][1])**2 <= r**2
                in_min_circle = (y-min_circle_center[i][0]) **2 + (x-min_circle_center[i][1])**2 <= r**2
                if in_square or in_max_circle or in_min_circle:
                    coords.append((y,x))

    return coords   

def pred_error(pred_obs, obs)
    future_img = pred_obs
    future_img2 = future_img.cpu().detach().numpy()
    future_img2 = future_img2.squeeze()
    future_img2 = np.transpose(future_img2, (1,2,0))
    coords_future = np.argwhere(future_img2[:,:,0] > 0.1)
    
    coords = img_mask(5, obs)
    err = 0
    for f_c in coords_future:
        if tuple(f_c) not in coords:
            err += future_img2[f_c[0],f_c[1],0]
    return err

def eval_agent(env_test, model, ep=5):
    
    hist = {"collision":[],"timeout":[], "speed_ratio":[], "col_speed":[]}
    for i in range(ep):
        obs = env_test.reset()
        done = False
        speed_ratio = []
        if i%50==0:
            print(i)
        while not done:
            action = model.compute_action(obs)
            pred_obs = model.future_state_prediction(prev_obs)
            err = pred_error(pred_obs, obs)
            print(err)
            obs, rewards, done, info = env_test.step(action)
            speed_ratio.append(info["speed_ratio"])
            
            if done:
                hist["collision"].append(info["collision"])
                hist["col_speed"].append(info["col_speed"])
                hist["timeout"].append(info["timeout"])   
                hist["speed_ratio"].append(np.mean(speed_ratio))

    return hist
    
    

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

config = configparser.ConfigParser()
config.read("params_RL.ini")
config  = eval_ini(config)
#    config = {}

log_dir = "tmp/gym/"
os.makedirs(log_dir, exist_ok=True)


num_cpu = 1  # Number of processes to use

#    
#    gym.envs.register(
#      id='myenv-v0',
#      entry_point='simulation.env:TrainEnv',
#      max_episode_steps=10000,
#      kwargs={"obs_builder":SimpleObservationBuilder(), "conf":config}
#  )

env_id = 'myenv-v0'

policy_config = dqn.APEX_DEFAULT_CONFIG
policy_config["num_workers"] = 1
policy_config["buffer_size"] = 1000000



policy_config["env_config"] = config
policy_config["env_config"]["env"]["obs_builder"] = eval(policy_config["env_config"]["env"]["obs_builder"])()
policy_config["framework"] = "torch"
#    policy_config["run"]=dqn.ApexTrainer
policy_config["env"]=TrainEnv
policy_config["horizon"]=1200

policy_config["batch_mode"]= "complete_episodes"

#https://stats.stackexchange.com/questions/221402/understanding-the-role-of-the-discount-factor-in-reinforcement-learning#:~:text=The%20discount%20factor%20essentially%20determines,that%20produce%20an%20immediate%20reward.
policy_config["gamma"] = 0.97 
policy_config['train_batch_size']= 512

#policy_config["exploration_config"]["final_epsilon"] = 0.05
#policy_config["exploration_config"]["epsilon_timesteps"] = 50000

policy_config["seed"] = 43
policy_config["evaluation_interval"]= 2
policy_config["evaluation_duration"]= 30

policy_config["evaluation_duration_unit"]= "episodes"
policy_config['evaluation_config']= {'explore': False}
policy_config['evaluation_num_workers']= 2
#    policy_config['rollout_fragment_length']= 200
policy_config["model"]["custom_model"] = "LowresCNNLSTM_test"
policy_config['log_level']= 'ERROR'

env = TrainEnv(policy_config["env_config"])
#ModelCatalog.register_custom_model("LowresCNNLSTM_test", LowresCNNLSTM_test)
#model = LowresCNNLSTM_test( obs_space=env.observation_space, 
#                  action_space=env.action_space, num_outputs = 1, model_config=policy_config["model"], name="LowresCNNLSTM_test")


test_agent = dqn.ApexTrainer(env=TrainEnv, config=policy_config)
test_agent.restore("/home/aplissonneau/simulateur/rllib_test/3obs_img/9/checkpoint_003600/checkpoint-3600")


#############################


obs = env.reset()
action=test_agent.compute_action(obs)
obs_arr = []
pred_obs_arr = []
for i in range(10):
    obs_arr.append(obs)
    prev_obs = obs
    pred_obs_arr.append(model.future_state_prediction(prev_obs))
    obs, _, _, _ = env.step(action)
    action = test_agent.compute_action(obs)