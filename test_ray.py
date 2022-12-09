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
import matplotlib.pyplot as plt
import decision_tree.decision_tree_learning as dt



def eval_agent(env_test, model, ep=5, dt=False):

    hist = {"collision":[],"timeout":[], "speed_ratio":[], "col_speed":[], "reward":[], "n_step":[]}
    for i in range(ep):
        obs = env_test.reset()
        done = False
        speed_ratio = []
        cum_rew = 0
        n_step = 0
        if i%50==0:
            print(i)
        while not done:
            n_step += 1
            if not dt:
                action = model.compute_action(obs)
            else:
                action = model.compute_action(env_test.obstacles, env_test.train.coord, env_test.train.speed)
            obs, rewards, done, info = env_test.step(action)
            cum_rew += rewards
            speed_ratio.append(info["speed_ratio"])

            if done:
                hist["collision"].append(int(info["collision"]))
                hist["col_speed"].append(info["col_speed"])
                hist["timeout"].append(int(info["timeout"]))
                hist["speed_ratio"].append(np.mean(speed_ratio))
                hist["reward"].append(cum_rew)
                hist["n_step"].append(n_step)

    return hist

if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    from observation_builder.obs_builder import *
    from simulation.env import TrainEnv
    from callbacks import *
    from custom_policies_ray import *
    import cv2
    #    multiprocessing.freeze_support()
    import ray.rllib.agents.dqn as dqn
    import ray
    from ray import tune
    from ray.rllib.models import ModelCatalog
    import json
    import rllib_configs as configs
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--checkpoint', type=str,
                         default="rllib_test/3obs_img/CNN_LSTM_aux/checkpoint_003600/checkpoint-3600",
                        help='Checkpoint to test. This script will infers from this path the related params.json used for training')

    parser.add_argument('--show', action="store_true",
                         default=False,
                        help='Used to activate visualisation')
    parser.add_argument('--obs_num', type=int,
                         default=3,
                        help='Number of obstacles to use in the test')

    parser.add_argument('--num_ep', type=int,
                         default=1000,
                        help='Number of episodes used to test the agent')

    parser.add_argument('--decision_tree', type=bool,
                     default=False,
                    help='If true a decision tree will be used instead of the RL agent.')

    parser.add_argument('--decision_tree_params', type=str,
                     default="decision_tree/params.json",
                    help='The decision tree\'s parameters. Only used if decision_tree is true.')

    args = parser.parse_args()
    print(args)

    config_ray_file = os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)), "params.json")
    file=open(config_ray_file)
    policy_config = json.load(file)



    policy_config["num_workers"] = 1

    f = policy_config["env_config"]["env"]["obs_builder"]
    f = f.split(".")[2].split(" ")[0]
    policy_config["env_config"]["env"]["obs_builder"] = eval(f)()
    policy_config["framework"] = "torch"

    policy_config["env"]=TrainEnv

    c = policy_config["callbacks"]
    c = c.split(".")[1][:-2]
    policy_config["callbacks"]=eval(c)
    policy_config['log_level']= 'ERROR'

    s = policy_config["sample_collector"]
    s = s.split("'")[1]

    policy_config["sample_collector"] = eval(s)

    policy_config["env_config"]["render"]["render"] = args.show
    policy_config["env_config"]["obstacle"]["num_obstacle"] = args.obs_num

    policy_config["model"]["custom_model"] = eval(policy_config["model"]["custom_model"])

    if not args.decision_tree :
        test_agent = dqn.ApexTrainer(env=TrainEnv, config=policy_config)
        test_agent.restore(args.checkpoint)
    else :
        test_agent = dt.DT()
        test_agent.load_config(args.decision_tree_params)
        test_agent.learn_tree()

    env_test = TrainEnv(policy_config["env_config"])

    hist = eval_agent(env_test, test_agent, ep=args.num_ep, dt=args.decision_tree)
    #print(hist)

    print("collision mean:",np.mean(hist["collision"]))
    print("collision std:",np.std(hist["collision"]))
    print("collision min:",np.min(hist["collision"]))
    print("collision max:",np.max(hist["collision"]))


    print("timeout:",np.mean(hist["timeout"]))

    print("speed_ratio mean:",np.mean(hist["speed_ratio"]))
    print("speed_ratio std:",np.std(hist["speed_ratio"]))
    print("speed_ratio min:",np.min(hist["speed_ratio"]))
    print("speed_ratio max:",np.max(hist["speed_ratio"]))

    print("reward mean:",np.mean(hist["reward"]))
    print("reward std:",np.std(hist["reward"]))
    print("reward min:",np.min(hist["reward"]))
    print("reward max:",np.max(hist["reward"]))

    col_speed =  [i for i in hist["col_speed"] if i is not None]
    print("col speed mean:",np.mean(col_speed))
    print("col speed std:",np.std(col_speed))
    try:
        print("col speed min:",np.min(col_speed))
        print("col speed max:",np.max(col_speed))
    except:
        pass


    print("n step mean:", np.mean(hist["n_step"]))
    print("n step std:", np.std(hist["n_step"]))
    print("n step min:", np.min(hist["n_step"]))
    print("n step max:", np.max(hist["n_step"]))


    ################################################
    col_speed =  [i for i in hist["col_speed"] if i is not None]


    dict_res = {"collision mean":np.mean(hist["collision"]),
    "collision std":np.std(hist["collision"]),
#        "collision min":np.min(hist["collision"]),
#        "collision max":np.max(hist["collision"]),
    "collision":hist["collision"],

    "timeout mean":np.mean(hist["timeout"]),
    "timeout":hist["timeout"],

    "speed_ratio mean":np.mean(hist["speed_ratio"]),
    "speed_ratio std":np.std(hist["speed_ratio"]),
    "speed_ratio min":np.min(hist["speed_ratio"]),
    "speed_ratio max":np.max(hist["speed_ratio"]),
    "speed_ratio":hist["speed_ratio"],

    "reward mean":np.mean(hist["reward"]),
    "reward std":np.std(hist["reward"]),
    "reward min":np.min(hist["reward"]),
    "reward max":np.max(hist["reward"]),
    "reward":hist["reward"],

    "col speed mean":np.mean(col_speed),
    "col speed std":np.std(col_speed),
    "col speed min":np.min(col_speed) if col_speed else 0,
    "col speed max":np.max(col_speed) if col_speed else 0,
    "col_speed":hist["col_speed"],

    "n step mean":np.mean(hist["n_step"]),
    "n step std":np.std(hist["n_step"]),
    "n step min":np.min(hist["n_step"]),
    "n step max":np.max(hist["n_step"]),
    "n step":hist["n_step"]}


    # name = "run5_1000ep_1obs_t"
    # with open(f'analysis/validation/{name}.json', 'w') as fp:
        # json.dump(dict_res, fp,  indent=4)
