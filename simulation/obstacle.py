import numpy as np
import json
from simulation.elements import *
import random

manual_obs = False

depth = lambda L: isinstance(L, list) and max(map(depth, L))+1

class Obstacle():

    def __init__(self, num_obstacle=2, coord = None, target=None, manual_obs=False, obs = "2", verbose=False, env_size=100, **args):
        
        assert (coord is None) or (depth(coord)==2), "Coords shape needs to be [n,2] with n the nb of obstacles"
        
        self.num_obstacle = num_obstacle if coord is None else len(coord)
        self.env_size = env_size
        self.x_min = 35
        self.x_max = 30 #Relative to end position
        self.coord=coord
        self.target=target
        self.manual_obs = manual_obs
        self.min_speed = np.array([0] * num_obstacle)
        self.max_speed = np.array([2] * num_obstacle)
        self.speed = self.max_speed
        self.speed_reac = np.array([0.3] * num_obstacle)
        self.kill = False
        self.prev_coord = self.coord
        self.expired = False
        self.scenario_fn = None
        self.step_n = 0
        self.scenario = None
        self.verbose = verbose
        
    @property
    def manual_obs(self):
        print("getter method called")
        return self._manual_obs

     # a setter function
    @manual_obs.setter
    def manual_obs(self, manual_obs):
        self._manual_obs = manual_obs

        #not supported for the moment with the new obstacles subclasses.
        if self._manual_obs: ###!!! Getter setter
            pass
            """
            path_scen = self.scenario_fn

            with open(path_scen,"r") as f:
                self.scenario = json.load(f)

            coords = self.scenario[0]

            self.coord = np.array([coords[f"obs_{i}"] for i in range(len(coords))])
#            self.target = np.array([self.random_coord() if not i.all() else i for i in self.target])
            coords = self.scenario[1]
            self.target = np.array([coords[f"obs_{i}"] for i in range(len(coords))])
            """
        else:

            tmp = np.array([self.obs_trajectory_gen(self.coord) for i in range(self.num_obstacle)])
            self.coord = tmp[:,0]
            self.target = tmp[:,1]

    def random_coord(self):

        y = random.random() * 10 - 5 # y between -5 and 5
        x = random.random() * (self.env_size - self.x_max  - self.x_min) + self.x_min # x between 
        return [x,y]

    def obs_trajectory_gen(self, coord = None):
        if coord is None:
            coord = self.random_coord()
        target = self.random_coord()
        return [coord, target]
    
    def maj_coord(self, refresh_time, target = None):
        
        
        #Numpy version
        in_upper_bound_value = np.minimum(self.speed, self.max_speed)
        self.speed = np.maximum(in_upper_bound_value, self.min_speed)
        self.prev_coord=self.coord
        if target is not None:
                a = self.coord
                b = target
                rad = np.arctan2((b-a).T[1], (b-a).T[0])
                x1 = np.cos(rad) * (self.speed * refresh_time)
                y1 = np.sin(rad) * (self.speed * refresh_time)
                self.coord = np.array([a.T[0] + x1, a.T[1] + y1]).T
        else:
            self.coord.T[1] = self.coord.T[1] - (self.speed * refresh_time)
            

    def random_change(self, refresh_time):
        self.speed = self.speed + random.randint(-1,1) * self.speed_reac * refresh_time


#        self.target = [x2,y2]

    def step(self, refresh_time):

        self.step_n+=1

        self.random_change(refresh_time)

        self.maj_coord(refresh_time, target = self.target)

#        j=2
        dist=np.linalg.norm(self.target-self.coord,axis=1)
        for i in range(len(self.target)):
            if dist[i] < (self.speed[i] * refresh_time):
                if self._manual_obs:
                    try:
                        coords = self.scenario[self.step_n + 2]
                        self.target[i] = np.array([coords[f"obs_{i}"]])
#                        j+=1
                    except:
#                            print("EXPIRED OBS")
                        self.expired = True
                else:
                    _,self.target[i] = np.array(self.obs_trajectory_gen(self.coord[i]))


#                print("dt",dt)
        if self.verbose:
            print("Obs")


    def stop(self):
        self.kill = True




class Pieton(Obstacle):
    def __init__(self, num_obstacle=1, coord = None, obs = "2", verbose=False) :

        self.min_speed = np.array([0] * num_obstacle)
        self.max_speed = np.array([3] * num_obstacle)
        self.coord = [None,None] if coord is None else coord
        self.target = np.array([[None,None]] * 1)
        self.speed = self.max_speed
        self.speed_reac = np.array([0.3] * num_obstacle)

        Obstacle.__init__(self, num_obstacle, coord, obs, verbose)
