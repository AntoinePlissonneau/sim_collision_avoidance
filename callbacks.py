# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 12:58:32 2022

@author: aplissonneau

In this file can be defined any callback
"""

from typing import Dict
import argparse
import numpy as np

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf",
    help="The DL framework specifier.",
)
parser.add_argument("--stop-iters", type=int, default=2000)


class MyCallbacks(DefaultCallbacks):
    
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        episode.user_data["collision"] = False
        episode.hist_data["collision"] = []
        episode.user_data["timeout"] = False
        episode.hist_data["timeout"] = []
        episode.user_data["speed_ratio"] = []
        episode.hist_data["speed_ratio"] = []
        episode.user_data["reward_speed"] = []
        episode.hist_data["reward_speed"] = []
        
    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        collision = episode.last_info_for()["collision"]
        timeout = episode.last_info_for()["timeout"]
        speed_ratio = episode.last_info_for()["speed_ratio"]
        reward_speed = episode.last_info_for()["reward_speed"]

        episode.user_data["collision"] = collision
        episode.user_data["timeout"] = timeout
        episode.user_data["speed_ratio"].append(speed_ratio)
        episode.user_data["reward_speed"].append(reward_speed)
        
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.policy_config["batch_mode"] == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )

        episode.custom_metrics["collision"] = episode.user_data["collision"]
        episode.hist_data["collision"].append(episode.user_data["collision"])
        episode.custom_metrics["timeout"] = episode.user_data["timeout"]
        episode.hist_data["timeout"].append(episode.user_data["timeout"])
        episode.custom_metrics["speed_ratio"] = np.mean(episode.user_data["speed_ratio"])
        episode.hist_data["speed_ratio"] = episode.user_data["speed_ratio"]
        episode.custom_metrics["reward_speed"] = np.sum(episode.user_data["reward_speed"])
        episode.hist_data["reward_speed"] = episode.user_data["reward_speed"]
