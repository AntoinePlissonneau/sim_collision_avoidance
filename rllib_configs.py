# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:52:26 2022

@author: aplissonneau
"""

import ray.rllib.agents.dqn as dqn
from ray.rllib.utils import merge_dicts
from callbacks import MyCallbacks
from observation_builder.obs_builder import ImgObservationBuilder, SimpleObservationBuilder, ImgSeqObservationBuilder, ImgStackObservationBuilder
from simulation.env import TrainEnv
import custom_policies_ray


lr_start = 5e-4
lr_end = 5e-5
lr_end_time = 6e7


# Test config with low computational ressource requirements. 
# Used only for debugging or testing the code as it may leads to bad results of the model.
APEX_TEST_CONFIG = merge_dicts(
        dqn.APEX_DEFAULT_CONFIG, 
        {
                "num_workers" : 1,
                "buffer_size" : 500,
                "lr_schedule" : [[0, lr_start], [lr_end_time,lr_end], [1e8,5e-6]],
                "prioritized_replay_beta" : 0.2,
                "final_prioritized_replay_beta" : 0.2,
                "prioritized_replay_beta_annealing_timesteps" : 30000,
                "framework": "torch", 
                #https://stats.stackexchange.com/questions/221402/understanding-the-role-of-the-discount-factor-in-reinforcement-learning#:~:text=The%20discount%20factor%20essentially%20determines,that%20produce%20an%20immediate%20reward.
                "gamma" : 0.99,
                "train_batch_size": 16,
                "seed" : 43,
                "evaluation_interval": 5,
                "evaluation_duration": 100,
                "evaluation_duration_unit": "episodes",
                "evaluation_config":{ 
                        "explore": False
                        },
                "rollout_fragment_length": 200,
                "model" : {
                        "custom_model" : "LowresCNNLSTM"
                        },
                "log_level": "ERROR",
                "callbacks" : MyCallbacks,
                "env" : TrainEnv,
                "env_config": {
                        "train":{
                                "min_speed" : 0, 
                                "coord" : [0,0],
                                "base_speed" : 8, 
                                "speed" : 8,
                                "max_speed" : 8,
                                "speed_reac" : 2,
                                "target_speed" : 8,
                                "train_size" : [2,0.5]},
                        
                        "obstacle":{
                                "num_obstacle" : 1,
                                "min_speed" : 0,
                                "max_speed" : 2,
                                "speed" : 2,
                                "speed_reac" : 0.3,
                                "static" : False,
                                "bounds_x" : [20,90],
                                "bounds_y" : [-5,5]},
                        
                        "env":{
                                "timeout_step" : 4000,
                                "env_size" : 150,
                                "env_rate" : 10,
                                "is_action_acc" : True,
                                "discretize_action_space" : True,
                                "reward_speed_ratio" :  -0.001,
                                "reward_col_ratio" : -2,
                                "reward_timeout_ratio" : 0,
                                "reward_ended_ratio" : 1,
                                "collision_dist_x" : 4,
                                "collision_dist_y" : 0.5,
                                "collision_train_speed" : 0.15,
                                "save_states" : False,
                                "save_path" : ".",
                                "seed" : 1,
                                "obs_builder" : ImgSeqObservationBuilder(seq_len=2)},
                        
                        "render":{
                                "render" : False,
                                "render_x_bounds" : [0,100],
                                "render_y_bounds" : [-10,10],
                                "speed_factor" : 1}}})



















