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
    import rllib_configs as configs

    log_dir = "tmp/gym/"
    os.makedirs(log_dir, exist_ok=True)
    env_id = 'myenv-v0'
    
    policy_config = configs.APEX_TEST_CONFIG

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

    

