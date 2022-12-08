# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:47:09 2020

@author: aplissonneau
"""
import sys
from PyQt5 import QtGui, QtCore
from gui.gui import App

#filename = 'Model_knn_Decision_1.sav'
#loaded_model = pickle.load(open(filename, 'rb'))
#auto = True
#manual_obs = True
#REFRESH_TIME = 0.01 # Wael # passage de 0.25 à 0.05



if __name__ == '__main__':
    # Start instances

#        s = send(q)
#        s.start()
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtGui.QApplication(sys.argv)
    thisapp = App()
    thisapp.show()
    sys.exit(app.exec_())
