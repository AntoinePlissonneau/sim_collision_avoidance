# sim_collision_avoidance
[Readme in progress]

## Introduction

The purpose of this git is two-fold:

 - The first one is to provide the code and the data used for validation of our paper for reproductibility
 
 - The second one is to provide in open access the simulator 
 
## Credits

Antoine Plissonneau - Railenium / UPHF LAMIH
Luca Jourdan - Railenium

Copyright (c) 2022 IRT Railenium. All Rights Reserved.

Copyrights licensed under an Academic/non-profit use license.

See the accompanying LICENSE file for terms.

## Installation

The simplest way to install all the dependencies of this project is to use Anaconda and to create an environment with the environment.yml file: 

```bash
conda env create -f environment.yml
conda activate simu_col
```


Torch and Cuda versions are relative to your GPU setup and then may be different that the ones specified in the environment.yml. This requirement file works for Ubuntu 20.04.5 LTS + CUDA 11.4. If you have installation issues, feel free to contact us.


 ### Test our models
 This git allows to test already trained algorithms in the train obstacle avoidance simulator. Example of use:
 
```
$ python test_ray.py --checkpoint "rllib_test/3obs_img/CNN_LSTM_aux/checkpoint_003600/checkpoint-3600" --show --obs_num 3 --num_ep 1000
```

## Arguments
* `--checkpoint`: path to the checkpoint to test (`str`)
* `--show`: If used, display the test scenario
* `--obs_num`: Number of obstacles to use in test scenario
* `--num_ep`: Number of episode to test on

 ### Analyse the validation data
 
 
 ### Use your custom models

Several aspect are customizable


## Install

```bash
conda env create -f environment.yml
conda activate simu_col
```


Torch and Cuda versions are relative to your GPU setup and then may be different that the ones specified in the environment.yml. This requirement file works for Ubuntu 20.04.5 LTS + CUDA 11.4  




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
=======
## Analysis

The data and scripts used to compute the figures and the table presented in the paper are available in the "Analysis" folder. 


