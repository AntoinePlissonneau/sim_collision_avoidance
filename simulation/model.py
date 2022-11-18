# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 18:53:02 2021

@author: aplissonneau
"""
import inspect
import numpy as np
import collections
import pickle
import os
import pandas as pd
import time
#def select_nearest(obs:dict)-> dict:
#    dist_obs = [obs[f"obs_{i}"]["dist_train_obs"] if obs[f"obs_{i}"]['obs_front_of_train'] else np.inf for i in range(len(obs)) ]
##    dist_obs = [obs[f"obs_{i}"]["dist_train_proj"] + obs[f"obs_{i}"]["dist_rail"]**2 if obs[f"obs_{i}"]['obs_front_of_train'] else np.inf for i in range(len(obs)) ]
#
#    nearest_obs_id = np.argmin(dist_obs)
#    selected_obs = obs[f"obs_{nearest_obs_id}"]
#    return selected_obs


def select_nearest(obs:dict)-> dict:
    coeff=5
    dist_obs = [obs[f"obs_{i}"]["dist_train_proj"]**2 + (coeff * obs[f"obs_{i}"]["dist_rail"])**2 if obs[f"obs_{i}"]['obs_front_of_train'] else np.inf for i in range(len(obs)) ]
    nearest_obs_id = np.argmin(dist_obs)
    selected_obs = obs[f"obs_{nearest_obs_id}"]
    return selected_obs

def features_selection(obs:dict, features_names:list)->np.array:
    features = np.array([[obs[name] for name in features_names]], dtype=object)
    return features


class Model():
    def __init__(self, params):
        self.params = params
        self._model_name = params["model_name"]
        self.loaded_model = self.load_model(self.model_name)
#        if self.params["model_name"] != "safety_bag":
#            self.loaded_model = self.load_model(self.params["model_name"])
#        else: 
#            self.loaded_model = None
        self.pp_pipe = Pipeline(self.params["pp_params"])
        
    def __repr__(self):
        return self._model_name
        
    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        self._model_name = model_name
        self.loaded_model = self.load_model(model_name)
        print("Model_name: ", self._model_name)
        
#    @model_name.getter
#    def model_name(self):
#        return self._model_name
        
        
    def load_model(self,model_name):
        if model_name != "safety_bag.txt":
            return pickle.load(open(os.path.join("models",model_name), 'rb'))
        else: 
            return None
        
    
    def preprocessing(self, data):
        d_obs = {k:v for k,v in data.items() if k.startswith("obs")}
        features = self.pp_pipe.process(d_obs)
        features = np.array(features[0])
        return features
    
    def exception_handling(self, features, params):
        if self.params["model_name"] == "Model_xgb_Decision.sav":
            features= pd.DataFrame(features,
                                       columns = ['dist_rail_0', 'dist_train_proj_0', 'obs_speed_0'])
        if self.params["model_name"].startswith("Model_xgb_Decision_"):
            features= pd.DataFrame(features,
                                       columns = ['dist_rail', 'dist_rail_deriv', 'dist_train_proj', 'dist_train_proj_deriv', 'obs_speed'])
        if self.params["model_name"]== ("safety_bag.txt"):
                    features= pd.DataFrame(features,
                                               columns = ['dist_rail', 'dist_rail_deriv', 'dist_train_proj', 'dist_train_proj_deriv', 'obs_speed'])

        return features
    
    def predict(self,data):
        try:
            features = self.preprocessing(data)
        except:
            return 15
        features = self.exception_handling(features, self.params)
#        print(features)
#        if self.params["model_name"]
        if self._model_name == "safety_bag.txt":
            if (abs(features.dist_rail[0]) < 0.5) and (features.dist_train_proj[0] < 100) and (features.dist_train_proj[0] > 0):
                speed_pred = 0
            else:
                speed_pred = 15
        else:
            speed_pred = self.loaded_model.predict(features)[0]
        return speed_pred


class Pipeline():
    def __init__(self, params):
        self.params = params
        self.functions = [eval(p) for p in params]
        self.names = [f.__name__ + "(" + ",".join(inspect.getfullargspec(f).args) + ")" for f in self.functions]
        
        argsspecs_0 = inspect.getfullargspec(self.functions[0])
        self.input_info = {i:argsspecs_0.annotations[i] for i in argsspecs_0.args}
        argsspecs_last = inspect.getfullargspec(self.functions[-1])
        self.output_info = argsspecs_last.annotations["return"]
    
    def process(self, *data, trace=False):
        seed = data
        if trace:
            print("input ==>", data)
        for func in self.functions:
            seed = func(*seed,**self.params[func.__name__])
            seed = tuple([seed])
            if trace:
                print(func.__name__,"==>", seed)
        return seed
    
    def check(self):
        pass
    
    def __repr__(self):
        return " -> ".join(self.names)
    
    def __str__(self):
        return " -> ".join(self.names)
    
    def inspect_pipeline(self):
        pass
    
if __name__ == "__main__":
    import time
    
#    path_parent = os.path.dirname(os.getcwd())
#    os.chdir(path_parent)
    d = {'train_speed': 3, 't': 0.0, 'dt': 0,
 'obs_0': {'obs_speed': 3.0,
           'obs_on_rail': False,
           'proj_coord': [87.98826636788006, 0],
           'dist_rail': 4.61,
           'dist_train_proj': 87.84,
           'dist_train_obs': 87.9591888277719,
           'obs_front_of_train': True},
 'obs_1': {'obs_speed': 3.0, 'obs_on_rail': False, 'proj_coord': [58.58416413395697, 0], 'dist_rail': -0.81, 'dist_train_proj': 58.43, 'dist_train_obs': 58.43971038435994, 'obs_front_of_train': True}}
  
    params = {"model_name":"Model_xgb_Decision.sav",
              "pp_params":{
                      "select_nearest":{},
                      "features_selection":{
                                             "features_names":["dist_rail", "dist_train_proj", "obs_speed"]
                                             }
                      }}
    params = collections.OrderedDict(params)
    m = Model(params)
    t0 = time.time()
    for i in range(1000):
        speed = m.predict(d)
    print(time.time() - t0)
    
    data = {'train_speed': 3, 't': 17.33, 'dt': 0.0, 'obs_0': {'obs_speed': 2.864999999999999, 'obs_on_rail': False, 'proj_coord': [46.35248665719774, 0], 'dist_rail': -1.78, 'dist_train_proj': 12.0, 'dist_train_obs': 12.133911695845361, 'obs_front_of_train': True}, 'obs_1': {'obs_speed': 2.864999999999999, 'obs_on_rail': False, 'proj_coord': [58.751371047405584, 0], 'dist_rail': -1.53, 'dist_train_proj': 24.4, 'dist_train_obs': 24.44902041698427, 'obs_front_of_train': True}}
