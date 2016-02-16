"""Color based visualizer of 4 dimensions.

pass the csv to plot from the command line.

built to take csv's generated from spider
    so the last column should be the sensor
    the second to last column should be time
    beyond that, it takes only the first two columns.
        which means that columns after the first two, and before the last two, will be omitted.
"""

#TODO someday, a better version of this should be integrated with the spider system, so we can see graphs in mostly real time.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
import sys

def scale(x, index, low, high):
    #return an array after discarding points outside of the range.
    return x[:,(x[index]>low) & (x[index]<high)]#select column indices within the range.

def init(selForN):
    
    fig = plt.figure() #init figure
    ax = fig.add_subplot(111, projection='3d') #create subplot in figure.
    
    #scale by output dimension, color
    selForN = scale(selForN, index=-1, low=-0.01, high=0.008)
    
    #set the color map to the output dimension.
    #print selForN
    scatPlot = ax.scatter(selForN[0], selForN[1], selForN[-2], c=selForN[-1]) #create plot
    
    ax.set_xlabel("Muscle Length")
    ax.set_ylabel("Balance")
    ax.set_zlabel("Timesteps Behind DeltaX Value")
    
    #add colorbar label
    cb = fig.colorbar(scatPlot)
    cb.set_label("DeltaX")
    
    return fig, scatPlot

#data = np.genfromtxt('data/VB_fit.csv', delimiter=',') #gather 4d data from csv.
data = np.genfromtxt(sys.argv[-1], delimiter=',') #gather 4d data from csv passed from command line.

print data

#cull data to a set number of points. if data has less than 'size points, some will be repeated.
randRows = np.random.randint(np.ma.size(data, axis=0),size=1500)
data = data[randRows, :]

data = data.transpose() #transpose for easier numpy handling.

init(data)

plt.show()