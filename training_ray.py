# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:51:37 2021

@author: aplissonneau
"""
REFRESH_TIME = 0.1


if __name__ == '__main__':
    import os

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    import ray.rllib.agents.dqn as dqn
    import ray
    from ray import tune
    from callbacks import MyCallbacks
    import custom_policies_ray
    from rllib_configs import *
    import argparse

    parser = argparse.ArgumentParser(description='Train the agent')

    parser.add_argument('--config', type=str,
                         default="APEX_TEST_CONFIG",
                        help='Configuration dict')
    parser.add_argument('--checkpoint_freq', type=int,
                         default=150,
                        help='Checkpoint save frequency')
    

    args = parser.parse_args()
    print(args)

    log_dir = "tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    env_id = 'myenv-v0'
    
    policy_config = eval(args.config)

    ray.init()
    tune.run(
        dqn.ApexTrainer,
        name="3obs_img",
        stop={"agent_timesteps_total": 250000000},
        config=policy_config,
        local_dir="rllib_test",
        checkpoint_at_end=True,
        checkpoint_freq=150,
        #restore=""
    )

    

