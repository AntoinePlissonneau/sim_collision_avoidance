from simulation.elements import *


class Perception():

    def __init__(self, env:Environment, verbose=False):
        #class instance, list of class instance
        self.train = env.train
        self.obstacles = env.obstacle
        self.previous_state = None
        self.scenario = []
        self.verbose=verbose


    def dist_obst_rail(self):
        train = {"train_speed":self.train.speed}#, "t":round(self.t,2), "dt":round(self.dt,4)

        obstacle = {f"obs_{i}": {
                "obs_speed":self.obstacles[i].speed,
                "obs_on_rail":np.abs(self.obstacles[i].coord[1]) <= 0.5,
                "proj_coord":[self.obstacles[i].coord[0],0],
#                "dist_rail": round(self.obstacles.coord[i][1],2),
                "dist_rail": abs(round(self.obstacles[i].coord[1],2)),

                "dist_rail_deriv": abs(round(self.obstacles[i].coord[1],2)) - self.previous_state[f"obs_{i}"]["dist_rail"] if self.previous_state is not None else 0,

                "dist_train_proj":round(self.obstacles[i].coord[0] - self.train.coord[0],2),
                "dist_train_proj_deriv":round(self.obstacles[i].coord[0] - self.train.coord[0],2) - self.previous_state[f"obs_{i}"]["dist_train_proj"] if
                    self.previous_state is not None else 0,
                "dist_train_obs": math.sqrt((self.obstacles[i].coord[0] - self.train.coord[0])**2 + (self.obstacles[i].coord[1] - self.train.coord[1])**2),
#                "dist_train_obs_deriv": math.sqrt((self.obstacles.coord[i][0] - self.train.coord[0])**2 + (self.obstacles.coord[i][1] - self.train.coord[1])**2) - self.previous_state[f"obs_{i}"]["dist_train_obs"] if self.previous_state is not None else 0,
                "obs_front_of_train": round(self.obstacles[i].coord[0] - self.train.coord[0],2) > 0
                } for i in range(len(self.obstacles))}

        features = {**train,**obstacle}
        return features

    def step(self):

        features = self.dist_obst_rail()
        self.previous_state = features

        if self.verbose:
            print("Per")

        return features

    def set_obstacle(self, obstacles):
        self.obstacles = obstacles
        self.previous_state = None
        self.step()

class Perception_probabiliste(Perception):

    def dist_obst_rail(self):
        train = {"train_speed":self.train.speed}#, "t":round(self.t,2), "dt":round(self.dt,4)

        obstacle = {f"obs_{i}": {
                "obs_speed":self.obstacles[i].speed,
                "obs_on_rail":np.abs(self.obstacles[i].coord[1]) <= 0.5,
                "proj_coord":[self.obstacles[i].coord[0],0],
#                "dist_rail": round(self.obstacles.coord[i][1],2),
                "dist_rail": abs(round(self.obstacles[i].coord[1],2)),

                "dist_rail_deriv": abs(round(self.obstacles[i].coord[1],2)) - self.previous_state[f"obs_{i}"]["dist_rail"] if self.previous_state is not None else 0,

                "dist_train_proj":round(self.obstacles[i].coord[0] - self.train.coord[0],2),
                "dist_train_proj_deriv":round(self.obstacles[i].coord[0] - self.train.coord[0],2) - self.previous_state[f"obs_{i}"]["dist_train_proj"] if
                    self.previous_state is not None else 0,
                "dist_train_obs": math.sqrt((self.obstacles[i].coord[0] - self.train.coord[0])**2 + (self.obstacles[i].coord[1] - self.train.coord[1])**2),
#                "dist_train_obs_deriv": math.sqrt((self.obstacles.coord[i][0] - self.train.coord[0])**2 + (self.obstacles.coord[i][1] - self.train.coord[1])**2) - self.previous_state[f"obs_{i}"]["dist_train_obs"] if self.previous_state is not None else 0,
                "obs_front_of_train": round(self.obstacles[i].coord[0] - self.train.coord[0],2) > 0
                } for i in range(len(self.obstacles))}

        features = {**train,**obstacle}
        return features
