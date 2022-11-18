# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 19:20:02 2021

@author: aplissonneau
"""


import numpy as np
import math
import torch
from gym import spaces

class ObservationBuilder():
    
    def __init__(self):
        self.previous_state = None
        self.observation_space = None
        self.highest_value = None
        self.lowest_value = None
        self.dtype = None
    def compute_observation(self):
        features = None
        return features


class SimpleObservationBuilder(ObservationBuilder):
    def __init__(self, ):
        super().__init__()        
        self.observation_space = (4,)
        self.highest_value = np.array([5,100,3,8])
        self.lowest_value = np.array([-5,-100,0, 0])
        self.dtype = np.float32
        
        self.observation_space = spaces.Box(
                                    low=self.lowest_value,
                                    high=self.highest_value,
                                    shape=self.observation_space,
                                    dtype=self.dtype)

    def select_nearest(self, obs:dict)-> dict:
        dist_obs = [obs[f"obs_{i}"]["dist_train_obs"] if obs[f"obs_{i}"]['obs_front_of_train'] else np.inf for i in range(len(obs)-1) ]
        nearest_obs_id = np.argmin(dist_obs)
        selected_obs = obs[f"obs_{nearest_obs_id}"]
        return selected_obs
    
    def compute_observation(self,env):
        self.train = env.train
        self.obstacles = env.obstacles

        features = self.dist_obst_rail()
        selected_obs = self.select_nearest(features)
        obs = [selected_obs["dist_rail"], selected_obs["dist_train_proj"], selected_obs["obs_speed"], self.train.speed]
        return obs
    
    def dist_obst_rail(self):
        train = {"train_speed":self.train.speed}#, "t":round(self.t,2), "dt":round(self.dt,4)
#        print("TRAIN",self.train.coord[0])
#        print(round(float(self.train.coord[0]),2))
        obstacle = {f"obs_{i}": {
                "obs_speed":self.obstacles.speed[i],
                "obs_on_rail":np.abs(self.obstacles.coord[i][1]) <= 0.5,
                "proj_coord":[self.obstacles.coord[i][0],0],
#                "dist_rail": round(self.obstacles.coord[i][1],2),
                "dist_rail": abs(round(self.obstacles.coord[i][1],2)),
                "dist_rail2": round(self.obstacles.coord[i][1],2),
#                "dist_rail_deriv": abs(round(self.obstacles.coord[i][1],2)) - self.previous_state[f"obs_{i}"]["dist_rail"] if self.previous_state is not None else 0,

                "dist_train_proj":round(self.obstacles.coord[i][0] - float(self.train.coord[0]),2),
#                "dist_train_proj_deriv":round(self.obstacles.coord[i][0] - self.train.coord[0],2) - self.previous_state[f"obs_{i}"]["dist_train_proj"] if self.previous_state is not None else 0,
                "dist_train_obs": math.sqrt((self.obstacles.coord[i][0] - float(self.train.coord[0]))**2 + (self.obstacles.coord[i][1] - float(self.train.coord[1]))**2),
#                "dist_train_obs_deriv": math.sqrt((self.obstacles.coord[i][0] - self.train.coord[0])**2 + (self.obstacles.coord[i][1] - self.train.coord[1])**2) - self.previous_state[f"obs_{i}"]["dist_train_obs"] if self.previous_state is not None else 0,
                "obs_front_of_train": round(self.obstacles.coord[i][0] - float(self.train.coord[0]),2) > 0
                } for i in range(self.obstacles.num_obstacle)}

        features = {**train,**obstacle}
        return features    
    

    

class ImgObservationBuilder(ObservationBuilder):
    def __init__(self, window=[[-10, -5], [60, 5]], step =[1,1]):
        super().__init__()
        self.window = np.array(window)
        self.step = np.array(step)
        self.n_channels = 3
        img_size = (self.window[1,:] - self.window[0,:])/self.step

        self.observation_space = (self.n_channels, int(img_size[1]), int(img_size[0]))
        self.highest_value = np.zeros(self.observation_space,dtype=np.uint8) + 255
        self.lowest_value = np.zeros(self.observation_space,dtype=np.uint8)
        self.dtype = np.uint8

        self.observation_space = spaces.Dict({"img":spaces.Box(
                                                        low=self.lowest_value,
                                                        high=self.highest_value,
                                                        shape=self.observation_space,
                                                        dtype=self.dtype),
                                            "features":spaces.Box(low=0.,
                                                       high=1.,
                                                       shape = (2,),
                                                       dtype = np.float32)})
    def compute_observation(self,env):
        self.train = env.train
        self.obstacles = env.obstacles

        features = self.dist_obst_rail()
        
        img = self.img_builder(features, self.window, self.step)
        return {"img":img, 
                "features":np.array([features["train_speed"]/self.train.max_speed, min(1,self.train.coord[0]/env.env_size)])}
    
    def dist_obst_rail(self):
        train = {"train_speed":self.train.speed}

        obstacle = {f"obs_{i}": {
                "dist_rail2": round(self.obstacles.coord[i][1],2),
                "dist_train_proj":round(self.obstacles.coord[i][0] - float(self.train.coord[0]),2),
                } for i in range(self.obstacles.num_obstacle)}

        features = {**train,**obstacle}
        return features    
        
    def img_builder(self, d, window, step):
        
        img_dim = ((window[1] - window[0])/step).astype(int)
        
        img = np.zeros([img_dim[1], img_dim[0], 3],dtype=np.uint8)
        
        point = np.array([[d[k]["dist_train_proj"], d[k]["dist_rail2"]] for k,v in d.items() if k.startswith("obs")])
        is_in = (point > window[0]) & (point < window[1])
        point2 = point[is_in.all(1)] - window[0]
    
        grid_spacing = step
        index = np.floor(point2 / grid_spacing).astype(int)
    #    print(index)
        for ind in index:
            img[img.shape[0] - ind[1]-1, ind[0],0] = 255
            
        #  Rails channel
        dist_inter_rail = 1
        img_y_center = img.shape[0]/2
        l_bound = img_y_center - 1/2*dist_inter_rail/step[1]
        l_idx = int(np.round(l_bound))
        u_bound = img_y_center + 1/2*dist_inter_rail/step[1]
        u_idx = int(np.round(u_bound))
    
        img[l_idx:u_idx,:,1] = 255
        
        # Train channel
        
        img[l_idx:u_idx, int(-window[0][0]/step[0]), 2] = 255
    
        img = np.rollaxis(img, 2, 0)

        return img
    



class ImgSeqObservationBuilder(ObservationBuilder):
    def __init__(self, window=[[-10, -5], [60, 5]], step =[1,1], seq_len = 4, slide = 1):
        super().__init__()
        self.window = np.array(window)
        self.step = np.array(step)
        self.n_channels = 3
        img_size = (self.window[1,:] - self.window[0,:])/self.step
        self.seq_len = seq_len
        self.slide = slide
        self.img_seq = np.zeros((4,3,10,70))
        self.counter = 0
        self.history_seq_len = (self.seq_len - 1) * slide +1
        self.history_seq = np.zeros((self.history_seq_len,3,10,70))

        self.observation_space = (self.seq_len, self.n_channels, int(img_size[1]), int(img_size[0]))
        self.highest_value = np.zeros(self.observation_space,dtype=np.uint8) + 255
        self.lowest_value = np.zeros(self.observation_space,dtype=np.uint8)
        self.dtype = np.uint8


        self.observation_space = spaces.Tuple((spaces.Box(
                                                        low=self.lowest_value,
                                                        high=self.highest_value,
                                                        shape=self.observation_space,
                                                        dtype=self.dtype),
                                            spaces.Box(low=0.,
                                                       high=1.,
                                                       shape = (2,),
                                                       dtype = np.float32)
                                                    ))


    def reset(self):
        self.history_seq = np.zeros((self.history_seq_len,3,10,70))
    def compute_observation(self,env):
        self.train = env.train
        self.obstacles = env.obstacles
        
        features = self.dist_obst_rail()
        img = self.img_builder(features, self.window, self.step)
        img = np.expand_dims(img, 0)
        self.history_seq = np.append(self.history_seq, img, axis=0)
        self.history_seq = np.delete(self.history_seq,0, 0)

#        return {"img":self.history_seq[::self.slide], 
#                "features":np.array([features["train_speed"]/self.train.max_speed])}
        return (self.history_seq[::self.slide], 
                np.array([features["train_speed"]/self.train.max_speed, min(1,self.train.coord[0]/env.env_size)]))
                
                
                
    def dist_obst_rail(self):
        train = {"train_speed":self.train.speed}

        obstacle = {f"obs_{i}": {
                "dist_rail2": round(self.obstacles.coord[i][1],2),
                "dist_train_proj":round(self.obstacles.coord[i][0] - float(self.train.coord[0]),2),
                } for i in range(self.obstacles.num_obstacle)}
        features = {**train,**obstacle}
        return features  
        
    def img_builder(self, d, window, step):
        
        img_dim = ((window[1] - window[0])/step).astype(int)
        
        img = np.zeros([img_dim[1], img_dim[0], 3],dtype=np.uint8)
        
        point = np.array([[d[k]["dist_train_proj"], d[k]["dist_rail2"]] for k,v in d.items() if k.startswith("obs")])
        is_in = (point > window[0]) & (point < window[1])
        point2 = point[is_in.all(1)] - window[0]
    
        grid_spacing = step
        index = np.floor(point2 / grid_spacing).astype(int)
    #    print(index)
        for ind in index:
            img[img.shape[0] - ind[1]-1, ind[0],0] = 255
            
        #  Rails channel
        dist_inter_rail = 1
        img_y_center = img.shape[0]/2
        l_bound = img_y_center - 1/2*dist_inter_rail/step[1]
        l_idx = int(np.round(l_bound))
        u_bound = img_y_center + 1/2*dist_inter_rail/step[1]
        u_idx = int(np.round(u_bound))
    
        img[l_idx:u_idx,:,1] = 255
        
        # Train channel
        
        img[l_idx:u_idx, int(-window[0][0]/step[0]), 2] = 255
    
        img = np.rollaxis(img, 2, 0)

        return img


class ImgSeqAuxObservationBuilder(ObservationBuilder):
    def __init__(self, window=[[-10, -5], [60, 5]], step =[1,1], seq_len = 4, slide = 1):
        super().__init__()
        self.window = np.array(window)
        self.step = np.array(step)
        self.n_channels = 3
        img_size = (self.window[1,:] - self.window[0,:])/self.step
        self.seq_len = seq_len
        self.slide = slide
        self.img_seq = np.zeros((4,3,10,70))
        self.counter = 0
        self.history_seq_len = (self.seq_len - 1) * slide +1
        self.history_seq = np.zeros((self.history_seq_len,3,10,70))

        self.observation_space = (self.seq_len, self.n_channels, int(img_size[1]), int(img_size[0]))
        self.highest_value = np.zeros(self.observation_space,dtype=np.uint8) + 255
        self.lowest_value = np.zeros(self.observation_space,dtype=np.uint8)
        self.dtype = np.uint8

#        self.observation_space = spaces.Dict({"img":spaces.Box(
#                                                        low=self.lowest_value,
#                                                        high=self.highest_value,
#                                                        shape=self.observation_space,
#                                                        dtype=self.dtype),
#                                            "features":spaces.Box(low=0.,
#                                                       high=1.,
#                                                       shape = (1,),
#                                                       dtype = np.float32)})

        self.observation_space = spaces.Tuple((spaces.Box(
                                                        low=self.lowest_value,
                                                        high=self.highest_value,
                                                        shape=self.observation_space,
                                                        dtype=self.dtype),
                                            spaces.Box(low=0.,
                                                       high=1.,
                                                       shape = (3,),
                                                       dtype = np.float32),
                                            spaces.Repeated(spaces.Box(2,), max_len=10)
                                                    ))


    def reset(self):
        self.history_seq = np.zeros((self.history_seq_len,3,10,70))
    def compute_observation(self,env):
        self.train = env.train
        self.obstacles = env.obstacles
        
        features = self.dist_obst_rail()
        img = self.img_builder(features, self.window, self.step)
        img = np.expand_dims(img, 0)
        self.history_seq = np.append(self.history_seq, img, axis=0)
        self.history_seq = np.delete(self.history_seq,0, 0)

#        return {"img":self.history_seq[::self.slide], 
#                "features":np.array([features["train_speed"]/self.train.max_speed])}
        return (self.history_seq[::self.slide], 
                np.array([features["train_speed"]/self.train.max_speed, min(1,self.train.coord[0]/env.env_size)], self.train.coord[0]),
                aux_task)
                
                
                
    def dist_obst_rail(self):
        train = {"train_speed":self.train.speed}

        obstacle = {f"obs_{i}": {
                "dist_rail2": round(self.obstacles.coord[i][1],2),
                "dist_train_proj":round(self.obstacles.coord[i][0] - float(self.train.coord[0]),2),
                } for i in range(self.obstacles.num_obstacle)}
        aux_task = [self.step * (self.obstacles.coord[i]//self.step) for i in range(self.obstacles.num_obstacle)]
        features = {**train,**obstacle}
        return features  
        
    def img_builder(self, d, window, step):
        
        img_dim = ((window[1] - window[0])/step).astype(int)
        
        img = np.zeros([img_dim[1], img_dim[0], 3],dtype=np.uint8)
        
        point = np.array([[d[k]["dist_train_proj"], d[k]["dist_rail2"]] for k,v in d.items() if k.startswith("obs")])
        is_in = (point > window[0]) & (point < window[1])
        point2 = point[is_in.all(1)] - window[0]
    
        grid_spacing = step
        index = np.floor(point2 / grid_spacing).astype(int)
    #    print(index)
        for ind in index:
            img[img.shape[0] - ind[1]-1, ind[0],0] = 255
            
        #  Rails channel
        dist_inter_rail = 1
        img_y_center = img.shape[0]/2
        l_bound = img_y_center - 1/2*dist_inter_rail/step[1]
        l_idx = int(np.round(l_bound))
        u_bound = img_y_center + 1/2*dist_inter_rail/step[1]
        u_idx = int(np.round(u_bound))
    
        img[l_idx:u_idx,:,1] = 255
        
        # Train channel
        
        img[l_idx:u_idx, int(-window[0][0]/step[0]), 2] = 255
    
        img = np.rollaxis(img, 2, 0)

        return img
    def discretized_position(self, features):
        
        step * (1.7//step)
        
class ImgStackObservationBuilder(ObservationBuilder):
    def __init__(self, window=[[-10, -5], [60, 5]], step =[1,1], seq_len = 4, slide = 1):
        super().__init__()
        self.window = np.array(window)
        self.step = np.array(step)
        self.n_channels =  2 + seq_len
        img_size = (self.window[1,:] - self.window[0,:])/self.step
        self.seq_len = seq_len
        self.slide = slide

        self.observation_space = (self.n_channels, int(img_size[1]), int(img_size[0]))
        self.highest_value = np.zeros(self.observation_space,dtype=np.uint8) + 255
        self.lowest_value = np.zeros(self.observation_space,dtype=np.uint8)
        self.dtype = np.uint8

        self.observation_space = spaces.Tuple((spaces.Box(
                                                        low=self.lowest_value,
                                                        high=self.highest_value,
                                                        shape=self.observation_space,
                                                        dtype=self.dtype),
                                            spaces.Box(low=0.,
                                                       high=1.,
                                                       shape = (2,),
                                                       dtype = np.float32)))

        
        img_dim = ((self.window[1] - self.window[0])/self.step).astype(int)
        
        img = np.zeros([img_dim[1], img_dim[0], 2],dtype=np.uint8)
        
        #  Rails channel
        dist_inter_rail = 1
        img_y_center = img.shape[0]/2
        l_bound = img_y_center - 1/2*dist_inter_rail/step[1]
        l_idx = int(np.round(l_bound))
        u_bound = img_y_center + 1/2*dist_inter_rail/step[1]
        u_idx = int(np.round(u_bound))
    
        img[l_idx:u_idx,:,0] = 255
        
        # Train channel
        
        img[l_idx:u_idx, int(-window[0][0]/step[0]), 1] = 255
    
        self.img = np.rollaxis(img, 2, 0)
        #self.img = self.img_.copy()
        self.obs_ = np.zeros([seq_len, img_dim[1], img_dim[0]],dtype=np.uint8)
        self.obs = self.obs_.copy()
        
    def reset(self):
        self.obs = self.obs_.copy()
        
    def compute_observation(self,env):
        self.train = env.train
        self.obstacles = env.obstacles
        
        features = self.dist_obst_rail()
        obs = self.img_builder(features, self.window, self.step)
        self.obs = np.append(self.obs, obs, axis=0)
        self.obs = np.delete(self.obs, 0, 0)
        
        sliced_obs = self.obs[::self.slide]
        
        img = np.concatenate([self.img, sliced_obs], axis = 0)
#        return {"img":self.history_seq[::self.slide], 
#                "features":np.array([features["train_speed"]/self.train.max_speed])}

        return (img, 
                np.array([features["train_speed"]/self.train.max_speed, min(1,self.train.coord[0]/env.env_size)]))
                
                
                
    def dist_obst_rail(self):
        train = {"train_speed":self.train.speed}

        obstacle = {f"obs_{i}": {
                "dist_rail2": round(self.obstacles.coord[i][1],2),
                "dist_train_proj":round(self.obstacles.coord[i][0] - float(self.train.coord[0]),2),
                } for i in range(self.obstacles.num_obstacle)}

        features = {**train,**obstacle}
        return features    
        
    def img_builder(self, d, window, step):
        
        img_dim = ((window[1] - window[0])/step).astype(int)
        
        img = np.zeros([img_dim[1], img_dim[0], 1],dtype=np.uint8)
        
        point = np.array([[d[k]["dist_train_proj"], d[k]["dist_rail2"]] for k,v in d.items() if k.startswith("obs")])
        is_in = (point > window[0]) & (point < window[1])
        point2 = point[is_in.all(1)] - window[0]
    
        grid_spacing = step
        index = np.floor(point2 / grid_spacing).astype(int)

        for ind in index:
            img[img.shape[0] - ind[1]-1, ind[0],0] = 255
            
        obs = np.rollaxis(img, 2, 0)
        return obs















