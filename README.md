# sim_collision_avoidance


## Description

The purpose of this git is two-fold:

 - The first one is to provide the code and the data used for validation of our paper for reproductibility
 
 - The second one is to provide in open access the simulator 
 
 
 ### Test our models
 This git allows to test already trained algorithms in the train obstacle avoidance simulator.
 
 ### Analyse the validation data
 
 
 ### Use your custom models

Several aspect are customizable


## Install

'''bash
conda env create -f environment.yml
conda activate simu_col
pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
'''

Torch and Cuda versions are relative to your GPU setup and then may be different that the ones specified in the environment.yml.




## Experiments configuration

Anyone can use configs created in rllib_config.py or add a new one in the same file. These configs are called on the main scripts.

## Training a model

python training_ray.py

## Testing a model

python test_ray.py


## Project organisation / create your custom agent

- Obs_builder.py : 
- ...
- ...