# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:32:45 2020

@author: aplissonneau
"""
import os

if __name__ == '__main__':
    os. chdir('..') #Pour absolute path tant que l'on a pas transformé en library
from copy import copy
import sys
import time
import pandas as pd
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
from threading import Thread, Lock
from queue import Queue
from PyQt5.QtWidgets import *#(QWidget, QToolTip, QDesktopWidget, QPushButton, QApplication)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QCoreApplication, QObject, QRunnable, QThread, QThreadPool, pyqtSignal, pyqtSlot 
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from simulation.env import TrainEnv
from observation_builder.obs_builder import ImgObservationBuilder
import json
import cv2
from simulation.env import TrainEnv
import rllib_configs as configs
import keyboard
import time



policy_config = configs.APEX_TEST_CONFIG
conf = policy_config["env_config"]
#conf["render"]["render"] = False


pg.setConfigOption('background', 'w')



class App(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)
        #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.generate_scenario = False
        #self.s = os.listdir("Experiments/scenarios/scenarios_rgv/")
        self.s_id = 0
        self.res=None
        self.stop_update = False
        self.env = None
        self.rail_len = 100
        def control_box():
            """Design a control box to set parameters, launch simulation and watch log
            """
            hbox = QHBoxLayout()
            vbox = QVBoxLayout()

            #File Box        
            self.file_run = QGroupBox("File choose")
            layout = QGridLayout()
            layout.setSpacing(10)
            map_file_label = QLabel("Map file: ")
    
            #Execution Log Box        
            self.text_group_box = QGroupBox("Execution log")
            layout_2 = QVBoxLayout()
            self.console = QTextEdit()
            self.console.setReadOnly(True)
            layout_2.addWidget(self.console)
            self.text_group_box.setLayout(layout_2)
    
    
            ## Obstacle mode option (random walk, manual)
            layout_obstacle_mode =  QHBoxLayout()
            label_obstacle_mode = QLabel("Obstacle mode: ")
            self.obstacle_mode = QComboBox()
            self.obstacle_mode.addItems(["Random Walk", "Manual"]) 
            layout_obstacle_mode.addWidget(label_obstacle_mode)
            layout_obstacle_mode.addWidget(self.obstacle_mode)
            layout_obstacle_mode.insertSpacing(-1,100)
            
            layout_obstacle_num =  QHBoxLayout()
            label_obstacle_num = QLabel("Obstacle num: ")
            self.obstacle_num = QSpinBox()
            self.obstacle_num.setRange(1, 10) 
            self.obstacle_num.setValue(2)
            #self.obstacle_mode
            layout_obstacle_num.addWidget(label_obstacle_num)
            layout_obstacle_num.addWidget(self.obstacle_num)
            layout_obstacle_num.insertSpacing(-1,100)
            
            
            ## Train speed
            layout_train_speed =  QHBoxLayout()
            label_train_speed = QLabel("Train max speed:")
            self.train_speed = QSpinBox()
            self.train_speed.setRange(1, 15)
            self.train_speed.setValue(2)
            self.train_speed.setMaximumWidth(100)
            layout_train_speed.addWidget(label_train_speed)
            layout_train_speed.addWidget(self.train_speed)
            layout_train_speed.insertSpacing(-1,100)

            ## Obstacle speed
            layout_obstacle_speed =  QHBoxLayout()
            label_obstacle_speed = QLabel("Obstacle max speed:")
            self.obstacle_speed = QSpinBox()
            self.obstacle_speed.setRange(1, 10)
            self.obstacle_speed.setValue(3)
            self.obstacle_speed.setMaximumWidth(100)
            layout_obstacle_speed.addWidget(label_obstacle_speed)
            layout_obstacle_speed.addWidget(self.obstacle_speed)
            layout_obstacle_speed.insertSpacing(-1,100)
            
            layout_realism_checkbox =  QHBoxLayout()
            self.realism_checkbox = QCheckBox('Dynamics check') 
            self.realism_checkbox.setChecked(1)
            layout_realism_checkbox.addWidget(self.realism_checkbox)
            
            layout_3 = QVBoxLayout()
            layout_3.addLayout(layout_obstacle_mode)
            layout_3.addLayout(layout_obstacle_num)
            layout_3.addLayout(layout_train_speed)
            layout_3.addLayout(layout_obstacle_speed)
            layout_3.addLayout(layout_realism_checkbox)

    
            self.comp = QGroupBox("Simulation parameters")

            self.comp.setLayout(layout_3)
    #        self.competitor_line.setDisabled(True)
    

            self.save_checkbox = QCheckBox('Save') 

            
            self.run_btn = QPushButton("Start", self)
            if self.save_checkbox.isChecked() and self.env is not None:
                self.run_btn.clicked.connect(self.save)
            self.run_btn.clicked.connect(self.run)

            layout.addWidget(self.save_checkbox, 1, 0, 1, 1)

            layout.addWidget(self.run_btn, 2, 0, 1, 4)
            #layout.setContentsMargins(0,0,0,0)
            layout.setVerticalSpacing(0)
            layout.setHorizontalSpacing(0)
            self.file_run.setLayout(layout)
    
            vbox.addWidget(self.comp)
            vbox.addWidget(self.file_run)
            vbox.addWidget(self.text_group_box)
    
            hbox.addLayout(vbox)
            return hbox
 
#        self.mainbox.setLayout(QtGui.QVBoxLayout())
        def graph_box():
            vbox_2 = QVBoxLayout()
    
    
            self.canvas = pg.GraphicsLayoutWidget()
            vbox_2.addWidget(self.canvas)
    
            self.label = QtGui.QLabel()
            vbox_2.addWidget(self.label)

            self.label2 = QtGui.QLabel()
            vbox_2.addWidget(self.label2)
            
            self.label3 = QtGui.QLabel()
            vbox_2.addWidget(self.label3)

            #  line plot
            self.otherplot = self.canvas.addPlot()
            self.otherplot.setXRange(0, self.rail_len, padding=0)
            self.otherplot.setYRange(-10, 10, padding=0)
    
            self.h3 = self.otherplot.plot(symbol = None)
            self.h3.setData([0,1000],[-0.5,-0.5])
            
            self.h4 = self.otherplot.plot(symbol = None)
            self.h4.setData([0,1000],[0.5,0.5])
            for i in range(250):
                self.otherplot.plot(symbol = None).setData([i*2,i*2], [-0.5,0.5])
    
#            self.h2 = self.otherplot.plot(symbol = ["o","t","t"], pen=None)


            return vbox_2
        
        vbox = QVBoxLayout()
        
        hbox = control_box()
        
        hbox_2 = QHBoxLayout()
        hbox_2.addLayout(hbox)

        self.vbox_2 = graph_box()
        hbox_2.addLayout(self.vbox_2)
        vbox.addLayout(hbox_2,5)
        
        vbox_3 = QVBoxLayout()
        self.canvas2 = pg.GraphicsLayoutWidget()
        vbox_3.addWidget(self.canvas2)
        self.otherplot2 = self.canvas2.addPlot()
        self.otherplot2.addLegend(size=(1,1),offset=(0,0))

        self.h4 = self.otherplot2.plot(symbol = None, pen=pg.mkPen("k", width=3), name="Commande")
        self.h5 = self.otherplot2.plot(symbol = None, pen=pg.mkPen("b", width=3), name="Consigne")
        self.otherplot2.setLabel("left","Speed")
        vbox.addLayout(vbox_3,1)
        
        self.mainbox.setLayout(vbox)



#        img = pg.ImageItem(np.random.normal(size=(100,10)))
#        img.scale(0.2, 0.1)
#        img.setZValue(-100)
#        pg.InfiniteLine(0, angle=0)
#        self.mainbox.layout().addItem(img)
        #### Set Data  #####################


        self.counter = 0
        self.fps = []
        self.lastupdate = time.time()
        self.speed_list = []
        self.target_list = []
        self.distance_list = []
        self.timer = time.time()

        #### Start  #####################
#        self._update()
    def save_scenario(self, scenario, path):
        try:
            scenario_id = len(os.listdir(path))
            scenario_path = os.path.join(path, f"scenario_{scenario_id+1}.json")
            with open(scenario_path, 'w') as outfile:
                json.dump(scenario, outfile)

        except Exception as error:
            print(error)

    def save(self, evaluation_data=False):
        
        idx = str(time.strftime("%Y%m%d-%H%M%S"))
        if self.pilot.auto:                    
            idx += "_auto_" + self.model_selection.currentText().__str__()# Wael # sauvegarde d'un fichier unique avec l'identifiant (temps de sauvegarde)

        if self.obstacle_mode.currentText().__str__() == "Manual":
            idx += f"_scenario{self.s_id}"

#        self.res={"col": collision,
#            "time_perf": np.mean(self.env.perf)}


        try:

            file_path = os.path.dirname(os.path.dirname(__file__))
            experiment_path = os.path.join(file_path, "Experiments")
            res_path = os.path.join(experiment_path, f"results_{idx}.csv")
            
            dico_meta = {"obs_mode": self.obstacle_mode.currentText().__str__(),
                         "pilot_mode": self.driver_mode.currentText().__str__(),
                         "model":self.model_selection.currentText().__str__(),
                         "filename":f"results_{idx}.csv",
                         "datetime": str(time.strftime("%Y%m%d-%H%M%S")),
                         "max_speed_train":self.env.train.max_speed,
                         "max_speed_obstacle":self.obstacle_speed.value(),
                         "collision" : str(self.env.col),
                         "num_obstacle": self.obstacle.num_obstacle}
            meta_path = os.path.join(experiment_path, f"results_{idx}.meta")
            with open(meta_path, 'w') as outfile:
                json.dump(dico_meta, outfile)
            
#                score_path = os.path.join(experiment_path, f"results_{idx}.res")
#                with open(score_path, 'w') as outfile:
#                    json.dump(self.res, outfile)
            
            
            pd.DataFrame(self.pilot.hist).to_csv(res_path)
            
        except Exception as error:
            print("Error while saving: ",error)
            pass

    def run(self):
        

        conf["obstacle"]["num_obstacle"] = self.obstacle_num.value()
        conf["env"]["env_rate"] = 20
        self.env_rate = conf["env"]["env_rate"]
        print(conf)
        self.env = TrainEnv(conf)
        
        self.env.reset()
        self.console.append(f"Simulation launched with parameters: \n" + \
                            f"   Obstacle: {self.obstacle_mode.currentText().__str__()}")


        try:
            self.h2.clear()
        except:
            pass
        self.h2 = self.otherplot.plot(symbol = ["s",*["o" for i in range(self.obstacle_num.value())]], 
                                    pen=None, 
                                    symbolSize=[20,*[10 for i in range(self.obstacle_num.value())]],
                                    symbolBrush=('b'))


        self.otherplot2.setYRange(0, 3, padding=0)
        
        self.i = 0
        self.tx = time.time()

        self._update()

        
    def _update(self):
        if self.stop_update:
            return
        
        
        action = 1
        if keyboard.is_pressed('z'):
            action = 2
        
        if keyboard.is_pressed('e'):
            action = 0
            
        self.env.step(action)
    
        
        self.data_train = self.env.train.coord
        self.data_obs = self.env.obstacles.coord

        nearest_obs_id = 0
        nearest_obs_value = self.data_obs[nearest_obs_id]
        try:
            self.data_obs = np.delete(self.data_obs,nearest_obs_id,0)
            self.data_obs = np.insert(self.data_obs, 0,nearest_obs_value,0)
        except Exception as error:
            print(error)
#            selected_obs = obs[f"obs_{nearest_obs_id}"]
#            return selected_obs

        datas = np.array([self.data_train, *self.data_obs])
#        print(datas)
#        self.img.setImage(self.data)
        self.h2.setData(datas.T[0],datas.T[1])
        if self.env.col:
            pass
            print("COLLISION found")
        expired = False
        if expired:
            print("Expired")
        if self.env.train.coord[0] >= self.rail_len or (self.env.col and not self.generate_scenario) or expired:
            print("DO?NE")
            
            self.run()
            return
        
        self.label.setText(f"Refresh rate: {self.env.env_rate}")

        self.label2.setText(f"Train speed: {round(self.env.train.speed,2)}")
#        self.label3.setText(f"Obstacle speed: {round(self.obstacle.speed,1)}") #Don't work with multiple obstacles
        self.counter += 1
        t2 = time.time()
        dt = t2 - self.tx
        slp = (self.i+1)/self.env_rate - dt
        if slp > 0:
            time.sleep(slp)
        print(slp)
        self.i+=1

        QtCore.QTimer.singleShot(1, self._update)

    def closeEvent(self,event):
        self.env.stop()
#        self.pilot.stop()
        self.stop_update = True
        
if __name__ == '__main__':
    # Start instances

#        s = send(q)
#        s.start()
    app = QtGui.QApplication(sys.argv)
    thisapp = App()
    thisapp.show()
    sys.exit(app.exec_())


