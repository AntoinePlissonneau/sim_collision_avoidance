# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:50:26 2020

@author: aplissonneau
"""

import time
from threading import Thread, Lock, Barrier
import math
import numpy as np
import os



auto = True
REFRESH_TIME = 0.1 # Wael # passage de 0.25 à 0.05
ENV_RATE = 10
#PER_RATE


mode_realism = True
SAVE_EACH = True


b1 = Barrier(2)
#q_train = Queue()
#q_obs = Queue()
#q_per = Queue()
#q_action = Queue()
REPLAY_BUFFER = []
lock = Lock()


"""
min_speed = 0 
coord = [0,0]
base_speed = 3 
speed = %(base_speed)s
max_speed = 3 
"""
class Train():

    def __init__(self, verbose=False, **args):
        self.coord = [0.,0.]
        self.base_speed = args.get("base_speed",3.) # = 2/s
        self.min_speed = args.get("min_speed",0)
        self.max_speed = args.get("max_speed",3.)

        self.speed = self.base_speed
#        self.speed_reac = 10
        self.kill = False
        self.target_speed = self.base_speed
#        self.train_size = [2,0.5] #[longueur_du_train, largeur_du_train]
#        self.train_range = [[self.coord[0] - self.train_size[0], self.coord[0]] ,
#                             -self.train_size[1], self.train_size[1]]
        self.prev_coord=self.coord
        self.step_n = 0 #Check synchronization
        self.verbose = verbose

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, speed_value):
        if speed_value > self.max_speed:
            speed_value = self.max_speed

        if speed_value < self.min_speed:
            speed_value = self.min_speed

        self._speed = speed_value


    def maj_coord(self, refresh_time):

        self.prev_coord = self.coord
        self.coord[0] = self.coord[0] + (self.speed * refresh_time)

    def step(self, refresh_time):

        self.maj_coord(refresh_time)

        if self.verbose:
            print("Train")
        self.step_n += 1

    def reset(self):
        self.coord = [0,0]



if __name__ == "__main__":

#    import os
#
#    os.chdir("..")
#    train = Train(verbose=False)
##    train.start()
#    obstacle = Obstacle(num_obstacle=2, coord = None, verbose=False)
#
#    env = Environment(train, obstacle, env_rate=20)
#    per = Perception(env)
#    pilot = Pilots.Pilot(train, per, REFRESH_TIME)
#    env.start()
#    pilot.start()
    pass
