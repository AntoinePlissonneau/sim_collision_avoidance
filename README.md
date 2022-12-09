# sim_collision_avoidance
[Readme in progress]

## Introduction
This git project has been made on the context of our paper "Deep reinforcement learning with predictive
auxiliary task for train collision avoidance" submitted to the journal Transactions on Intelligent Transportation Systems (TITS) 
(this paper is at this moment under review). 

With this project, the ambition is not only to share the implementation of the works made for this paper, but also to offer the
community an easy-to-use and efficient tool for future work on train collision avoidance. 
Thus, it is easily modular in several aspects. First, in addition to already implemented train dynamics, anyone can
easily incorporate their own train or obstacle dynamics. An observation builder comes as an external wrapper of the environment,
allowing to directly build its own state representation (e.g. tabular, image-like, . . . ). In the same way, constructing a
custom reward function is facilitated by the architecture. The simulation can be run in real time or in accelerated
time, allowing hours of driving to be simulated in minutes. A graphical visualization can also be activated for training,
and testing and to enable the possibility of driving the train manually with the keyboard, allowing, among other things, the
use of imitation learning. 

In this repo, you will find:

- A train collision avoidance simulation environment built on Gym.
- A manual driving script with a graphical interface enabling the possibility of driving the train
manually with the keyboard.
- The models presented in the paper with their best checkpoints (CNN, CNN-LSTM and CNN-LSTM with predictive auxiliary task and decision tree).
- The data used in the results chapter in the paper.
- Many tools to build custom policies, observation builder and train dynamics.
- A train script to train your own models
- A test script to validate our models or your own


 
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

## Train model

Anyone can use configs created in rllib_config.py or add a new one in the same file. These configs are used to setup the experiment. 

Train an agent with the command:

```
$ python training_ray.py
```

## Test models
This git allows to test already trained algorithms in the train obstacle avoidance simulator. Example of use:
 
```
$ python test_ray.py --checkpoint "rllib_test/3obs_img/CNN_LSTM_aux/checkpoint_003600/checkpoint-3600" --show --obs_num 3 --num_ep 1000
```

_Arguments_:
* `--checkpoint`: path to the checkpoint to test (`str`)
* `--show`: If used, display the test scenario
* `--obs_num`: Number of obstacles to use in test scenario
* `--num_ep`: Number of episode to test on

## Manual driving

```
$ python manual_driving.py
```

## Analyse the data used in the result section of our paper
The data and scripts used to compute the figures presented in the paper are available in the "Analysis" folder. 

 
## Build a custom agent

### Custom policy

```
$ custom_policies_ray.py
```

### Custom observation builder

```
$ observation_builder/obs_builder.py
```
### Custom dynamics

```
simulation/functions.py
```
```
simulation/env.py
```


