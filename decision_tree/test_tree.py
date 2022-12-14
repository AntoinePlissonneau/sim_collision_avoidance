REFRESH_TIME = 0.1

import warnings
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from rllib_configs import APEX_TEST_CONFIG
import argparse
import decision_tree.decision_tree_learning as dt

warnings.filterwarnings('ignore')


def generate_trajectories(env_test, model, nbr_ep):

    hist = {"collision":[],"nb_step":[], "speed_ratio":[], "col_speed":[], "reward":[]}
    for i in range(nbr_ep):

        obs = env_test.reset()
        done = False
        speed_ratio = []
        rewards = []
        j=0

        while not done:
            action = model.compute_action(env_test.obstacles, env_test.train.coord, env_test.train.speed)

            j+=1
            obs, reward, done, info = env_test.step(action)
            speed_ratio.append(info["speed_ratio"])
            rewards.append(reward)

            if done:
                hist["collision"].append(int(info["collision"]))
                try:
                    hist["col_speed"].append(float(info["col_speed"]))
                except:
                    hist["col_speed"].append(None)
                hist["nb_step"].append(int(j))
                hist["speed_ratio"].append(float(np.mean(speed_ratio)))
                hist["reward"].append(rewards)

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


if __name__ == '__main__':
    import configparser
    from simulation.env import TrainEnv
    import json

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--num_ep', type=int,
                         default=1000,
                        help='Number of episodes used to test the agent')

    parser.add_argument('--decision_tree_params', type=str,
                     default="decision_tree/params.json",
                     help='The decision tree\'s parameters. Only used if decision_tree is true. \
                        An exemple of parameters file is present in "decision_tree/params.json"')


    args = parser.parse_args()

    env_test = TrainEnv(APEX_TEST_CONFIG["env_config"])



    data = pd.read_csv("decision_tree/dataset.csv")
    data = data.dropna()

    X = data.loc[:,["obs_y","obs_x", "train_speed"]].to_numpy()

    Y = data.loc[:,["action"]].to_numpy()



    params = [5,"entropy",[4,1,3],0.8,12]

    model = dt.DT()
    model.load_config(args.decision_tree_params)
    model.learn_tree()


    env_test = TrainEnv(APEX_TEST_CONFIG["env_config"])

    hist = generate_trajectories(env_test, model, args.num_ep)



    print(hist["collision"])
    print("collision mean:",np.mean(hist["collision"]))
    print("collision std:",np.std(hist["collision"]))
    print("collision min:",np.min(hist["collision"]))
    print("collision max:",np.max(hist["collision"]))


    print("speed_ratio mean:",np.mean(hist["speed_ratio"]))
    print("speed_ratio std:",np.std(hist["speed_ratio"]))
    print("speed_ratio min:",np.min(hist["speed_ratio"]))
    print("speed_ratio max:",np.max(hist["speed_ratio"]))

    print("reward mean:",np.mean(hist["reward"]))
    #print("reward std:",np.std(hist["reward"]))
    #print("reward min:",np.min(hist["reward"]))
    #print("reward max:",np.max(hist["reward"]))

    col_speed =  [i for i in hist["col_speed"] if i is not None]
    print("col speed mean:",np.mean(col_speed))
    print("col speed std:",np.std(col_speed))
    try:
        print("col speed min:",np.min(col_speed))
        print("col speed max:",np.max(col_speed))
    except:
        pass


    print("n step mean:", np.mean(hist["nb_step"]))
    print("n step std:", np.std(hist["nb_step"]))
    print("n step min:", np.min(hist["nb_step"]))
    print("n step max:", np.max(hist["nb_step"]))
