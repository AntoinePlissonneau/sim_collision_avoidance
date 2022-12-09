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

Anyone can use the configurations available in ```rllib_config.py``` or add a new one in the same file (more details behind). These configs are used to setup the experiment. 
Be careful when selecting the "num_worker", "buffer_size" and "batch_size" values because it depend of your hardware capabilities.

Train an agent with the command:

```
$ python training_ray.py --config APEX_CNNLSTM_aux_CONFIG --checkpoint_freq 150
```

_Arguments_:
* `--config`: name of the config to use (`str`)
* `--checkpoint_freq`: The number of training steps before dumping the network parameters (`int`)

The checkpoints, callbacks and experiments params are saved in the folder ```rllib_test/3obs_img/```.


## Test models
This git allows to test already trained algorithms in the train obstacle avoidance simulator. Example of use:
 
```
$ python test_ray.py --checkpoint "rllib_test/3obs_img/CNN_LSTM_aux/checkpoint_003600/checkpoint-3600" --show --obs_num 3 --num_ep 1000
```

_Arguments_:
* `--checkpoint`: path to the checkpoint to test (`str`)
* `--show`: If used, display the test scenario
* `--obs_num`: Number of obstacles to use in test scenario (`int`)
* `--num_ep`: Number of episode to test on (`int`)

## Manual driving

You can manually drive the train using:
```
$ python manual_driving.py
```
 Use your keyboard to accelerate ("Z") or brake ("E").



## Analyse the data used in the result section of our paper
The data and scripts used to compute the figures presented in the paper are available in the "Analysis" folder. 

 
## Build a custom agent / Custom environment


### Config file

The config is defined in ```rllib_configs.py```. It list all the parameters to use for the training of the agent like:
- The environment parameters
- The rl algorithm to use and its hyperparameters
- The model to use to estimate de policy and its hyperparameters
- ...

You can base your config file on existing configs for several reinforcement learning algorithms implemented in rllib: [https://docs.ray.io/en/latest/rllib/rllib-algorithms.html]

The common parameters of a config file are described here: [https://docs.ray.io/en/latest/rllib/rllib-training.html#common-parameters]


### Custom policy


You can build your own policy class in ```custom_policies_ray.py```.


More information on how to build a custom ray policy: [https://docs.ray.io/en/latest/rllib/rllib-concepts.html].

Note that to train an agent using this policy, you have to register it at the end of the script. Example:
```
ModelCatalog.register_custom_model("LowresCNN", LowresCNN)
```
and you have to call it in your custom config.Example:
```
"model" : {
	  "custom_model" : "LowresCNN"
			},
```
### Custom observation builder

You can build your own observation builder to modify the state representation in input of your model by creating a class in ```observation_builder/obs_builder.py```.

You also have to call it in your custom config.

### Custom environment

#### Custom dynamics

Multiple train dynamics are already implemented in```simulation/functions.py```. You can had your own or modify an existing one in this script.

To change the train dynamic in the environment, manually change it in ```simulation/env.py```.

#### Custom reward
The weights of the reward can be set in the config file. To create new rewards, directly modify ```simulation/env.py```.

