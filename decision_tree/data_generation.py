import numpy as np
import os
import sys

os.chdir("../")
sys.path.insert(0, './')
retval = os.getcwd()
print(retval)

def generate_trajectories(env_test, model, nbr_ep=5):
    """
    Generate trajectories using a model.
    env_test : the environment to use
    model : the model to use to control the actions
    nbr_ep : the number of episode to simulate
    return : a dictionary containing for each step of each episodes the folowing
            data : collision : boolean
                    nbr_step : int
                    speed_ration : float
                    col_speed : float
                    data : table containing the environment as perceived by the model
    """

    hist = {"collision":[],"nb_step":[], "speed_ratio":[], "col_speed":[], "data":[]}
    for i in range(nbr_ep):
        obs = env_test.reset()
        done = False
        speed_ratio = []
        d = []
        nb_step = 0

        while not done:

            action = model.compute_action(obs)
            features = {
              "obs_y_1": env_test.obstacles.coord[0][1],
              "obs_x_rel_1": env_test.obstacles.coord[0][0] - float(env_test.train.coord[0]),
              "train_speed": env_test.train.speed,
              "train_x": env_test.train.coord[0],
              "action": int(action)
              }
            nb_step += 1
            obs, rewards, done, info = env_test.step(action)
            speed_ratio.append(info["speed_ratio"])
            d.append(features)
            if done:
                hist["collision"].append(int(info["collision"]))
                hist["col_speed"].append(info["col_speed"])
                hist["nb_step"].append(nb_step)
                hist["speed_ratio"].append(np.mean(speed_ratio))
                hist["data"].append(d)
    return hist


def json_to_csv(csv_file):
    """
    transform every json file in the current directory into a unique file in the csv format.
    """
    with open(csv_file, 'w+') as writer:
        writer = csv.writer(writer)
        writer.writerow(['obs_y', 'obs_x', 'train_speed', 'action'])
        #json_list = listdir("./dataset/")
        directory = "./decision_tree/"
        json_list = listdir("./" + directory)
        datas = []
        for f in json_list:
            f = open("./"  + directory + str(f))
            data = json.load(f)
            for i in range(len(data["data"])):
                episode = data["data"][i]
                collision = data["collision"][i]
                episode_size = len(episode)

                for j in range(episode_size) :
                    timestep = episode[j]
                    datas = []
                    try :

                        timestep["obs_x_rel_1"] = int(timestep["obs_x_rel_1"])
                        timestep["obs_y_1"] = int(timestep["obs_y_1"])

                        datas.append(timestep["obs_y_1"])
                        datas.append(timestep["obs_x_rel_1"])
                        datas.append(timestep["train_speed"])

                        if collision == 1 and j >= episode_size - 61:
                            datas.append(0)
                        else :
                            datas.append(timestep["action"])
                        writer.writerow(datas)
                    except :
                        print("there seem to be an error in one of the files.")



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

    test_agent = dqn.ApexTrainer(env=TrainEnv, config=policy_config)
    test_agent.restore(args.checkpoint)
    env_test = TrainEnv(policy_config["env_config"])


    hist = generate_trajectories(env_test, test_agent, 3)

    with open('./decision_tree/dataset.json', 'w') as f:
        json.dump(hist, f)

    json_to_csv('dataset.csv')
