from threading import Thread, Lock
from simulation.model import Model
import simulation.functions as f
import keyboard
import time
import collections
import os
import pandas as pd
from tensorforce.environments import Environment
from tensorforce.agents import Agent


lock = Lock()

class Pilot:

    def __init__(self, train, per, REFRESH_TIME, realism=True):
        Thread.__init__(self)
        self.train = train
        self.kill = False
        self.acc = 0
        self.per = per
        self.realism = realism
        self.step_n = 0
        self.action = None
        self.model = None
        self.save_hist = False
        self.hist = []
        self.refresh_time = REFRESH_TIME

#        try:
#            params = {"model_name":"Model_xgb_Decision_rgv.sav",
#                      "pp_params":{
#                              "select_nearest":{},
#                              "features_selection":{
#                                                     "features_names":['dist_rail', 'dist_rail_deriv', 'dist_train_proj', 'dist_train_proj_deriv', 'obs_speed']
#                                                     }
#                              }}
#            params = collections.OrderedDict(params)
#            self.model = Model(params)
#        except Exception as error:
#            print("PB:", error)


    def set_save_hist(self, save) :
        self.save_hist = save

    def ATO(self, target_speed):

        target_acc = (target_speed - self.train.speed )/self.refresh_time
        if target_acc > 0:
            max_acc = f.traction(self.train.speed) #fonction à passer dans le constructeur.
            self.acc = min(target_acc, max_acc)
            self.train.speed += self.acc * self.refresh_time
        else:
            min_acc = f.freinage(self.train.speed)
            self.acc = max(target_acc, min_acc)
            self.train.speed += self.acc * self.refresh_time


    def run(self):
        while True:
            if self.kill:
                print("Pilot stopped")
                time.sleep(0.1)
                break

            else :
                features = self.per.step()
                target_speed = self.command(features)

                if self.realism :
                    self.ATO(target_speed)
                elif target_speed is not None:
                    self.train.speed = target_speed

                self.action = self.acc

                if self.save_hist :
                    self.hist.append({**features, "action": self.action})

                time.sleep(self.refresh_time/10)
                if self.debug :
                    print("sleeping " + str(self.refresh_time/10) + " seconds")


    def stop(self):
        self.kill = True

    def command(self):
        pass
    def save(self):
        pass

    def load(self):
        pass

    def end_episode(self, collision=False):
        pass

    def save_state_action(self):
        pass


class Manual_pilot(Pilot):

    def command(self, _):
        bypass=False
        target_speed = None
        if keyboard.is_pressed('z'):  # if key 'z' is pressed
            target_speed = self.train.speed + f.traction_rgv(self.train.speed) * self.refresh_time
            target_speed = min(self.train.max_speed, target_speed)

        if keyboard.is_pressed('e'):  # if key 'e' is pressed
            target_speed = self.train.speed + f.freinage_rgv(self.train.speed) * self.refresh_time
            target_speed = max(self.train.min_speed, target_speed)

        if keyboard.is_pressed('d'):  # if key 'd' is pressed
            target_speed = self.train.max_speed
            bypass=True

        return target_speed


class Auto_pilot(Pilot):

    def command(self, X):
        return 35
        with lock:
            target_speed = max(min(self.model.predict(X), self.train.max_speed), self.train.min_speed)
        return target_speed



class Rl_pilot(Pilot):

    def __init__(self, train, per, REFRESH_TIME=0.1, realism=True, debug=False, training=True, loading=None):
        self.debug = debug
        self.nbr_pas = 0
        self.episode = 0
        self.training = training
        self.states = []
        self.actions = []
        self.accident = []
        self.rewards = []
        self.total_reward = 0
        super().__init__(train, per, REFRESH_TIME, realism=realism)


        path = "../Experiments/RL/test1"

        self.environment = Environment.create(environment='rl', max_episode_timesteps=9999, pilot=self)
        """self.agent = Agent.create(
            agent='tensorforce', environment=self.environment, update=64,
            optimizer=dict(optimizer='adam', learning_rate=1e-3),
            objective='policy_gradient', reward_estimation=dict(horizon=20, discount=0.99),
            exploration=0.1
        )
        """
        if not loading :
            self.agent = Agent.create(
                agent='dqn', environment=self.environment,
                exploration=0.1, batch_size=1, memory=9999, discount=0.5
            )
        else :
            self.load(loading)
            print(self.agent)
        self.environment.reset()



    def get_state(self):
        features = self.per.step()
        obstacles = []
        for i in range(10) :
            try :
                dist_train = features["obs_"+str(i)]["dist_train_proj"]
                dist_rail = features["obs_"+str(i)]["dist_rail"]
                obstacles.append([dist_train, dist_rail, 1])
            except :
                obstacles.append([-1, -1, 0])

        return obstacles

    def get_train_speed(self):
        features = self.per.step()
        return features["train_speed"]

    def command(self, _):
        self.nbr_pas += 1
        try:
            reward = self.environment.compute_reward(self.get_train_speed(), False)
            self.agent.observe(terminal=False, reward=reward)
            self.total_reward += reward
        except:
            pass

        states = {"obstacles":self.get_state(), "train_speed":self.get_train_speed()}
        actions = self.agent.act(states=states, independent=not(self.training))
        states, self.terminal, self.reward = self.environment.execute(actions=actions)

        if not self.training :
            self.states.append(states)
            self.actions.append(actions)

        if self.debug :
            print(states)
            print("-----------------")
            print("episode numéro : " + str(self.episode))
            print("nombre de pas : " + str(self.nbr_pas))
            print("vitesse cible : " + str(actions[0]))
            print("reward : " + str(self.reward))
            print("vitesse actuelle : " + str(self.get_train_speed()))
            print("-----------------")
        return actions[0]


    def stop(self):
        self.kill = True


    def end_episode(self, collision=False):
        reward = self.environment.compute_reward(self.get_train_speed(), collision)

        if self.debug and collision :
            print("nombre de pas depuis la dernière collision : " + str(self.nbr_pas))
            print("obstacle rencontrer. reward = " + str(reward))
        try :
            self.agent.observe(terminal=True, reward=reward)
            self.total_reward += reward
        except :
            if self.debug:
                print("impossible d'attribuer la reward")

        if collision :
            self.accident.append(self.nbr_pas)
            self.rewards.append(self.total_reward)
            print("épisode numéro : " + str(len(self.rewards)))
            print("evolution du nombre de pas avant accident : " + str(self.accident))
            print("evolution de la reward totale : " + str([int(round(i, 0)) for i in self.rewards]))
            self.total_reward = 0
            self.nbr_pas = 0

        self.environment.reset()
        self.episode += 1


    def save(self, filename=None):
        if self.debug :
            print("save")

        if not filename :
            filename = str(time.strftime("%Y%m%d-%H%M%S"))

        path = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(path, "Experiments")
        path = os.path.join(path, "RL")

        self.agent.save(path, filename, format="checkpoint", append="episodes")


    def load(self, filename):
        print("loading model...")
        path = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(path, "Experiments/RL/")

        self.agent = Agent.load(directory=path, filename=filename, format="checkpoint", environment=self.environment,
                    agent='dqn', exploration=0.1, batch_size=1, memory=9999, discount=0.5)

    def save_state_action(self, directory, filename):
        #save in a csv file a list of states and the action taken by the model for each states.
        #states : a list of states. actions : a list of action. directory : the directory in which the file will be saved. filename : filename of the file to save.
        dictionary = {'state': self.states, 'action': self.actions}
        dataframe = pd.DataFrame(dictionary)
        dataframe.to_csv(directory+filename)
