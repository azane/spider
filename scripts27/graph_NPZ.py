import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
import sys

"""called from command line
the .npz file referenced should be...an .npz file...
and should have two arrays. x.shape == (s,inputdimensions) and y.shape == (s,outputdimensions)
#also must use frameworkpython on osx, this command can be set up in the .bashrc file if using a virtualenv. see matplotlib faq for details.
the the 2nd argument should be a comma delimited list of x columns to plot

terminal e.g.: frameworkpython thisfile.py data.npz -1,3,2"""

def cull_range(x, y, low, high, i=-1):
    """takes two 2d arrays, but only culls by the last y column."""
    indices = np.where((y[:,i] >= low) & (y[:,i] <= high))
    return x[indices], y[indices]

def graph3x1y(x, y, xCols=[0,1,-1], yCol=-1, yLow=None, yHigh=None, fig=None, sbpltLoc=111, numPoints=1000, **kwargs):
    """Takes two 2d arrays, but only graphs the 
        3 columns of x, and 1 column of y."""
    #---<Error Catching>---
    assert x.shape[0] == y.shape[0], "x and y have a different number of points!" #check sample size.
    
    assert y.ndim == 2
    assert y.shape[1] >= 1  # at least one y column
    
    assert x.ndim == 2
    assert len(xCols) == 3, "There must be 3 xColumns listed for plotting."
    assert x.shape[1] >= len(xCols), "There aren't enough xColumns for this plotting function."
    
    #check indices, if it's out of range, bad!
    assert max(xCols) < x.shape[1]
    if min(xCols) < 0:
        assert (min(xCols)*-1) <= x.shape[1]
    
    assert yCol < y.shape[1]
    if yCol < 0:
        assert (yCol*-1) <= y.shape[1]
    #---</Error Catching>---
    
    #---<Culling>---
    #cull by range first, just in case this gets the numPoints down.
    if (yLow is not None) and (yHigh is not None):
        #TODO allow the output to be culled by only high or only low
        #exclude points outside of low/high range
        x, y = cull_range(x, y, yLow, yHigh, i=yCol)
    
    #get set number of points.
    if x.shape[0] > numPoints:
        randRows = np.random.randint(x.shape[0],size=numPoints)
        x = x[randRows]
        y = y[randRows]
    #---</Culling>---
    
    #---<Plotting>---
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(sbpltLoc, projection='3d') #create subplot in figure.
    
    #set the color map to the output dimension.
    scatPlot = ax.scatter(
                            x[:,xCols[0]],
                            x[:,xCols[1]],
                            x[:,xCols[2]],
                        c=y[:,yCol], **kwargs) #create plot
    
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    
    #add colorbar label
    cb = fig.colorbar(scatPlot)
    cb.set_label("y")
    #---</Plotting>---
    
    return fig, [scatPlot, ax]


if __name__ == '__main__':
    
    #verify xCols
    xCols = sys.argv[2].split(',')
    try:
        xCols = [int(i) for i in xCols]
    except ValueError:
        raise ValueError("sys.argv[2] should be a comma delimited list of x column indices (ints) to plot.")
    
    #get data from npz
    xy = np.load(sys.argv[1])
    x = xy['x']
    try:
        y = xy['y']
    except KeyError:
        y = xy['t']
    
    print "x.shape: " + str(x.shape)
    print "y.shape: " + str(y.shape)
    
    #TODO write a function that takes y and x shapes, and returns a list of sbpltLoc's so they can all fit on one graph.
    #       then iterate it and add subplots to the first figure. or...use subplots and figures. figs for y, subplts for xdims over 3
    #sbpltLoc = 
    
    graph3x1y(x, y, xCols=xCols, yCol=-1, yLow=None, yHigh=None, fig=None, sbpltLoc=111, numPoints=1000)
    
    plt.show()