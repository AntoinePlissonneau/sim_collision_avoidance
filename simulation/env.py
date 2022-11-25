# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:12:51 2022

@author: aplissonneau
"""
import gym
import numpy as np
from simulation.elements import Train
from simulation.obstacle import Obstacle
import copy
from observation_builder.obs_builder import SimpleObservationBuilder, ImgObservationBuilder, ImgSeqObservationBuilder, ImgStackObservationBuilder
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib import patches
import simulation.functions as f

class TrainEnv(gym.Env):
    """OpenAI gym"""
    metadata = {'render.modes': ['human']}
    def __init__(self, env_config):
        super(TrainEnv, self).__init__()
#        self.train = Train()
#        self.obstacles = Obstacle(num_obstacle=1) # , static = [40,0]
        self.config = env_config
#        self.current_step = 0
#        self.collision = False
#        self.num_obstacle = num_obstacle
        self.reward_range = (-1, 0)

        self.render_b = self.config["render"].get("render",False)


        conf_env = self.config.get("env",{})
        self.obs_builder = conf_env.get("obs_builder",SimpleObservationBuilder())

        self.env_rate = conf_env.get("env_rate",10)
        self.timeout_step = conf_env.get("timeout_step",10000)
        self.reward_speed_ratio = conf_env.get("reward_speed_ratio", -1e-4)
        self.reward_col_ratio = conf_env.get("reward_col_ratio", -0.1)
        self.reward_timeout_ratio = conf_env.get("reward_timeout_ratio", 0)
        self.reward_ended_ratio = conf_env.get("reward_ended_ratio", 1)
        self.collision_dist_x = conf_env.get("collision_dist_x", 4)
        self.collision_dist_y = conf_env.get("collision_dist_y", 0.5)
        self.collision_train_speed = conf_env.get("collision_train_speed", 0.1)
        self.env_size = conf_env.get("env_size", 100)
        self.is_action_acc = conf_env.get("is_action_acc",True)
        self.discretize_action_space = conf_env.get("discretize_action_space",True)
        
        
        if self.discretize_action_space:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(
                low=np.array([-1]), high=np.array([1]), dtype=np.float32)

        
        self.observation_space = self.obs_builder.observation_space
        

        self.prev_obs=None                     
            

        self.n_step = 0
        
        
        if self.render_b:
            self.render_speed_factor = 4
            xmin = 0
            xmax = self.env_size
            fig, self.ax = plt.subplots()

            plt.ion()
            render_x_bounds = [0,self.env_size]
            render_y_bounds = [-5,5]
            plt.xlim(render_x_bounds)
            plt.ylim(render_y_bounds)
            
            plt.hlines(0.5, xmin, xmax)
            plt.hlines(-0.5, xmin, xmax)
            for i in range(xmin,xmax,int((xmax-xmin) /20)):
                plt.vlines(i, -0.5, 0.5)
        
            plt.pause(2)

    def reset(self):
    	# Reset the state of the environment to an initial state
        config_train = copy.copy(self.config.get("train",{}))
        self.train = Train(**config_train)
        self.min_n_step = self.env_size/(self.train.max_speed/self.env_rate)
        self.obstacles = Obstacle(**self.config.get("obstacle",{}), env_size=self.env_size)
#        self.collision = False
        self.current_step = 0
        if isinstance(self.obs_builder, ImgSeqObservationBuilder) or isinstance(self.obs_builder, ImgStackObservationBuilder):
            self.obs_builder.reset()
            
        if self.render_b:
            try:
                self.rec.remove()

                self.train_coord.remove()
                self.train_speed.remove()

                self.obstacle_coord.remove()
            except Exception as error:

                self.train_coord = plt.scatter(self.train.coord[0], 0, c = "blue")
                self.train_speed = plt.text(1, 1, str(round(self.train.speed,1)) +"/s")   
                self.obstacle_coord = plt.scatter(*self.obstacles.coord.T, marker = "d", c = "red")
                self.rec = patches.Rectangle((self.train.coord[0], self.train.coord[1] - self.collision_dist_y),
                                             self.collision_dist_x, self.collision_dist_y*2, alpha=0.3,color="orange")
                self.ax.add_patch(self.rec)
    
        return self._next_observation()


    def _next_observation(self):
        self.obs = self.obs_builder.compute_observation(self)
        return self.obs

    def is_collision(self, previous_dist_x, actual_dist_x, actual_dist_y):
        collisions = (previous_dist_x >= 0) & (actual_dist_x <= 3) & (self.train.speed > 0.15) & (abs(actual_dist_y) < 0.5) # Changement temporaire pour qu'il y ait plus de stimulation
        self.col = collisions.any()
    
    def step(self, action):
    	# Execute one time step within the environment
        self._take_action(action)

        # Update train position
        self.train.step(1/self.env_rate)
                
        # Compute and update obs position
        self.obstacles.step(1/self.env_rate)

        #Collision detection
        dists = self.obstacles.coord - self.train.coord
        actual_dist_x = dists[:,0]
        actual_dist_y = dists[:,1]
        previous_dist_x = actual_dist_x
        self.is_collision(previous_dist_x, actual_dist_x, actual_dist_y)


        timeout = self.current_step > self.timeout_step
        ended = self.train.coord[0] > self.env_size
        done = ended or self.col or timeout # collision to define
        self.prev_obs = self.obs
        self.obs = self._next_observation()
        #print("train speed 2:",self.train.speed)
        speed_ratio = self.train.speed / self.train.max_speed
        #print("ratio",speed_ratio)
#        reward_speed =  (speed_ratio **2)/self.min_n_step * self.reward_speed_ratio
        reward_speed =  (1 - (speed_ratio **(3/4))) * self.reward_speed_ratio
        #reward_speed =  self.reward_speed_ratio        
        reward_col = self.reward_col_ratio * self.col



        reward_timeout = self.reward_timeout_ratio * timeout
        reward_ended = ended*self.reward_ended_ratio
        reward = reward_speed + reward_col + reward_timeout + reward_ended
        if self.col:
            col_speed=self.train.speed / self.train.max_speed
        else:
            col_speed = None                            
        self.n_step+=1
        self.current_step += 1
        if self.render_b:
            self.render()
        
        return self.obs, float(reward), done, {"collision":self.col, "timeout":timeout, "reward_speed":reward_speed, "speed_ratio":speed_ratio, "col_speed":col_speed}

    def _take_action(self, action):
        
        if self.is_action_acc:
            if self.discretize_action_space:
                maintain_speed_action = int((self.action_space.n - 1)/2)
                if action == maintain_speed_action: # ensure there is a "do nothing" action
                    self.action = 0
                    min_action = 0
                    max_action = 0
                else:
                    #min_action = f.freinage_tram(self.train.speed * 3.6)
                    #max_action = f.traction_tram(self.train.speed * 3.6)
                    min_action = f.freinage_TFA(self.train.speed * 3.6)
                    max_action = f.traction_TFA(self.train.speed * 3.6)
                    norm_action = action / (self.action_space.n - 1) 
                    self.action = norm_action * (max_action -  min_action) + min_action
                
            else:

#                min_action = f.freinage_tram(self.train.speed * 3.6)
#                max_action = f.traction_tram(self.train.speed * 3.6)
                min_action = f.freinage_TFA(self.train.speed * 3.6)
                max_action = f.traction_TFA(self.train.speed * 3.6)
                norm_action = (action +1)/2
                self.action = norm_action * (max_action -  min_action) + min_action
            #print("action:",self.action)
            self.train.speed += self.action / self.env_rate
            #print("train_speed", self.train.speed)
        else:
            if self.discretize_action_space:
                self.action = action / (self.action_space.n - 1) 
            else:
                self.action = ((action +1)/2)

        
            self.train.speed = self.action * self.train_max_speed

    def render(self, mode='human', close=False):
        
    	# Render the environment to the screen
        try:
            self.train_coord.remove()
            self.train_speed.remove()
    #        train_speed.remove()
            self.obstacle_coord.remove()
            self.rec.remove()
        except:
#            print("pass")
            pass
        self.train_coord = plt.scatter(self.train.coord[0], 0, c = "blue")
        self.train_speed = plt.text(1, 1, str(round(float(self.train.speed),1)) +"/s     "+ str(round(float(self.action),1)))
        
        self.obstacle_coord = plt.scatter(*self.obstacles.coord.T, marker = "d", c = "red")
        self.rec = patches.Rectangle((self.train.coord[0], self.train.coord[1] - self.collision_dist_y),
                                     self.collision_dist_x, self.collision_dist_y*2, alpha=0.3,color="orange")
        self.ax.add_patch(self.rec)

#        obstacle_speed = plt.text(1, 2, str(round(env.obstacles.speed,1)) +"/s")
        plt.pause((1/self.env_rate)/self.render_speed_factor)


if __name__ == "__main__":
    
    env = TrainEnv(ImgObservationBuilder, {})
    env.reset()
    env.step([1])


