# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:40:45 2019

@author: auref
"""
import numpy as np
import matplotlib.pyplot as plt

plt.close("all")
# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels


thetas = range(-10,11)
thetas = np.array(thetas)
thetas = thetas / (10 / (np.pi/4))
lengthA = []
lengthB = []
momentArmA = []
momentArmB = []
attachA = 0.2
attachB = 0.4
insertA = 0.2
insertB = 0.2
for theta in thetas:
     lengthA.append(np.sqrt(np.power(attachA, 2) + np.power(insertA, 2) + 2 * attachA * insertA * np.sin(theta))) 
     lengthB.append(np.sqrt(np.power(attachB, 2) + np.power(insertB, 2) + 2 * attachB * insertB * np.sin(-theta)))
     momentArmA.append(attachA * insertA * np.cos(theta) / lengthA[-1])
     momentArmB.append(attachB * insertB * np.cos(-theta) / lengthB[-1])
plt.figure('Muscle length when varying theta')
plt.plot(thetas, lengthA)
plt.plot(thetas, lengthB)
plt.title('Muscle length when varying theta')
plt.xlabel('Theta (rad)')
plt.ylabel('Muscle length (m)')
plt.legend(['muscle A','muscle B'])
plt.grid()
plt.figure('Moment arm when varying theta')
plt.plot(thetas, momentArmA)
plt.plot(thetas, momentArmB)
plt.title('Moment arm when varying theta')
plt.xlabel('Theta (rad)')
plt.ylabel('Moment arm length (m)')
plt.legend(['muscle A','muscle B'])
plt.grid()