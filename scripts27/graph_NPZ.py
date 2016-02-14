#TODO someday, a better version of this should be integrated with the spider system, so we can see graphs in mostly real time.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
import sys

"""called from command line
the .npz file referenced should be...an .npz file...
and should have two arrays. x.shape == (s,inputdimensions) and y.shape == (s,outputdimensions)
#also must use frameworkpython on osx, this command can be set up in the .bashrc file if using a virtualenv. see matplotlib faq for details.

terminal e.g.: frameworkpython thisfile.py data.npz"""

#TODO create an "if main" clause so this can be used and imported.

def scale(x, index, low, high):
    #return an array after discarding points outside of the range.
    return x[:,(x[index]>low) & (x[index]<high)]#select column indices within the range.

def init4d(x, y):
    assert y.ndim == 1
    assert x.ndim == 2
    #assert x.shape[0] == 3
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') #create subplot in figure.
    
    #set the color map to the output dimension.
    scatPlot = ax.scatter(x[0], x[1], x[-1], c=y) #create plot
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    #add colorbar label
    cb = fig.colorbar(scatPlot)
    cb.set_label("y")
    
    return fig, scatPlot

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

assert x.shape[0] == y.shape[0], "x and y have a different number of points!" #check sample size.


#cull data to a set number of points.
lim = 1500
if x.shape[0] > lim:
    randRows = np.random.randint(x.shape[0],size=1500)
    x = x[randRows]#, :]
    y = y[randRows]#, :]

for t in range(y.shape[1]):
    #x is only transposed so x.shape == (inputdimensions, s), while each y output is taken so y.shape == (s,)
    init4d(np.squeeze(x.transpose()), np.squeeze(y[:,t])) #select single output, transpose x to match now that y is a 1d row of shape (s,)

plt.show()