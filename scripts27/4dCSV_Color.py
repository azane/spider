import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np

def init(selForN):
    
    fig = plt.figure() #init figure
    ax = fig.add_subplot(111, projection='3d') #create subplot in figure.
    
    #scale by output dimension, color
    low=-0.01
    high=0.008
    selForN = selForN[(selForN[:,-1]>low) & (selForN[:,-1]<high)]#select those indices within the range.
    
    #set the color map to the sensor dimension.
    #print selForN
    scatPlot = ax.scatter(selForN[:,0], selForN[:,1], selForN[:,-2], c=selForN[:, -1]) #create plot
    
    ax.set_xlabel("Muscle Length")
    ax.set_ylabel("Balance")
    ax.set_zlabel("Timesteps Behind DeltaX Value")
    
    cb = fig.colorbar(scatPlot)
    cb.set_label("DeltaX")
    
    return fig, scatPlot

#data = np.genfromtxt('data/VB_fit.csv', delimiter=',') #gather 4d data from csv.
data = np.genfromtxt('data/VB_fit.csv', delimiter=',') #gather 4d data from csv.

print data

#cull data to 100 points.
randRows = np.random.randint(np.ma.size(data, axis=0),size=1500)


data = data[randRows, :]

init(data)

plt.show()