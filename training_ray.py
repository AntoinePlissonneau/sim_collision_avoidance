# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:51:37 2021

@author: aplissonneau
"""
REFRESH_TIME = 0.1


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
    import configparser
    from observation_builder.obs_builder import ImgObservationBuilder, SimpleObservationBuilder, ImgSeqObservationBuilder, ImgStackObservationBuilder
    from simulation.env import TrainEnv
    import ray.rllib.agents.dqn as dqn
    import ray
    from ray import tune
    from callbacks import MyCallbacks
    
    
    config = configparser.ConfigParser()
    config.read("params_RL.ini")
    config  = eval_ini(config)

    log_dir = "tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)


    num_cpu = 5  # Number of processes to use
    env_id = 'myenv-v0'
    
    policy_config = dqn.APEX_DEFAULT_CONFIG
    policy_config["num_workers"] = 1
    policy_config["buffer_size"] = 10000
    lr_start = 5e-4
    lr_end = 5e-5
    lr_end_time = 6e7
    policy_config["lr_schedule"] = [[0, lr_start], [lr_end_time,lr_end], [1e8,5e-6]]
    policy_config["prioritized_replay_beta"] = 0.2
    policy_config["final_prioritized_replay_beta"] = 0.2
    policy_config["prioritized_replay_beta_annealing_timesteps"] = 30000

    
    
    
    policy_config["env_config"] = config
    policy_config["env_config"]["env"]["obs_builder"] = eval(policy_config["env_config"]["env"]["obs_builder"])(seq_len=2)
    policy_config["framework"] = "torch"
    policy_config["env"]=TrainEnv

    
    #https://stats.stackexchange.com/questions/221402/understanding-the-role-of-the-discount-factor-in-reinforcement-learning#:~:text=The%20discount%20factor%20essentially%20determines,that%20produce%20an%20immediate%20reward.
    policy_config["gamma"] = 0.99
    policy_config['train_batch_size']= 256
    policy_config["seed"] = 43
    policy_config["evaluation_interval"]= 5
    policy_config["evaluation_duration"]= 100
    policy_config["evaluation_duration_unit"]= "episodes"
    policy_config['evaluation_config']= {'explore': False}
    policy_config['rollout_fragment_length']= 200
    policy_config["model"]["custom_model"] = "LowresCNNLSTM"
    policy_config['log_level']= 'ERROR'
    policy_config["callbacks"] = MyCallbacks

    ray.init()
    tune.run(
        dqn.ApexTrainer,
        name="3obs_img",
        stop={"agent_timesteps_total": 250000000},
        config=policy_config,
        local_dir="rllib_test",
        checkpoint_at_end=True,
        checkpoint_freq=150,
        #restore="/home/aplissonneau/simulateur/rllib_test/3obs_img/ApexTrainer_TrainEnv_7888c_00000_0_2022-06-05_11-47-11/checkpoint_001450/checkpoint-1450"
    )

    

