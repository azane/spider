

#TODO someday, a better version of this should be integrated with the spider system, so we can see graphs in mostly real time.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
import sys

def scale(x, index, low, high):
    #return an array after discarding points outside of the range.
    return x[:,(x[index]>low) & (x[index]<high)]#select column indices within the range.

def init3d(x, y):
    assert y.ndim == 1 #exactly 1 output
    assert x.ndim == 2 #exactly 2d
    assert x.shape[0] == 2 #exactly 2 inputs
    
    fig = plt.figure() #init figure
    ax = fig.add_subplot(111, projection='3d') #create subplot in figure.
    
    #set the color map to the output dimension.
    #print selForN
    scatPlot = ax.scatter(x[0], x[1], y, c=y) #create plot
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    #add colorbar label
    cb = fig.colorbar(scatPlot)
    cb.set_label("y")
    
    return fig, scatPlot

def init2d(x,y):
    assert x.ndim == 1
    assert x.shape == y.shape

    return plt.scatter(x, y)#, s=area, c=colors, alpha=0.5)

xy = np.load(sys.argv[1]) #gather 3d data split into x and y from npz passed from command line.
x = xy['x']
y = xy['y']

print x.shape, y.shape

assert x.shape[0] == y.shape[0], "x and y have a different number of points!"

#cull data to a set number of points.
lim = 1500
if x.shape[0] > lim:
    randRows = np.random.randint(x.shape[0],size=1500)
    x = x[randRows, :]
    y = y[randRows, :]

for t in range(y.shape[1]):
    init2d(np.squeeze(x.transpose()), np.squeeze(y[:,t])) #select single output, transpose x to match now that y is a 1d row of shape (s,)

plt.show()