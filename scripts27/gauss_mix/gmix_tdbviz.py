"""brainstomring

make a visualizer file.
we need to visualize:

#update whole plot on iteration, i.e. over input range for this iteration
- one 1:1 plot of all means over the input range. limited to 1d inputs, 1d output.
- one graph for each target dimension showing variance over input range. 1d input range.
- one graph of all mixing coefficients showing m over input range. 1d input range.

#tack info on the end, i.e. over time.
- a graph of the error gradient with respect to each variable (weights/biases). each variable has it's own graph. x-iterations, y=error gradient
    -it should display in order of net execution.
- a graph of each variable (weights/biases) actual values. this should display like the gradients, but separately.
- a graph of the loss over iterations
- a graph of the likelihood over iterations

#full plot update, but not over input range
- a plot of a sampling of the model given current status, this will require different functions for different dimensionality due to visualization techniques.
- if you're up to it, a color plot of the mixture's probability distribution. : )

-- some way to visualize many:many input outputs for all of these.


#so! we
"""
import matplotlib.pyplot as plt
import numpy as np
import gmix_sample_mixture as smpl

#----<helper functions>----
def sort_by_x(x, y):
    #x and y should be 1d at this point .shape == [s,]
    xy = np.column_stack((x,y))  # stack horizontally
    xy = xy[xy[:,0].argsort()]  # sort by the x column
    return xy[:,0], xy[:,1]  # re-separate x and y
    
def x1_yMany(x, y, xLabel='', yLabel='', title=''):
    
    #x.shape == [s,x]
    #y.shape == [s,t/y]
    
    x_ = x[:,0]  # select the 1st input dimension
    
    #iterate the output dimensions, plotting each line with the original x values.
    for outDim in range(y.shape[1]):
        y_ = y[:,outDim]
        x_, y_ = sort_by_x(x_, y_)
        plt.plot(x_, y_)
    
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    
#----</helper functions>----


#----<CTX functions>----
def mixing_coefficients(ctx, x, g):
    x1_yMany(x, g, xLabel='Input Range', yLabel='Relevance', title='Mixing Coefficients over X')
    
def means(ctx, x, u):
    #x.shape == [s,x]
    #u.shape == [s,g,t]
    
    #graph g lines on t graphs
    
    #get subplot rows/cols, make rectangle as close to a square as possible
    numSubplt = u.shape[2]
    sideSubplt = np.sqrt(numSubplt)
    rows = int(sideSubplt)
    if ((sideSubplt-rows) > 0):
        cols = rows + 1
    else:
        cols = rows
    
    #iterate t
    #   plot g lines, then write a subplot after g's are finished.
    for outDim in range(u.shape[2]):
        #make plot of all g's for this subplot.
        x1_yMany(x, u[:,:,outDim], xLabel='Input Range', yLabel='Output Range', title='Means')
        plt.subplot(rows, cols, (outDim+1))

def variances(ctx, x, v):
    x1_yMany(x, v, xLabel='Input Range', yLabel='Standard Deviation', title='Standard Deviation over X')

def sample(ctx, x, m, v, u):
    x, y = smpl.sample_mixture(x, m, v, u)
    plt.scatter(x, y)
    plt.title('Sampling of GMM')

def watch_loss(ctx, loss):
    #snagged from tdb viz example file.
    if not hasattr(ctx, 'loss_history'):
        ctx.loss_history=[]
    ctx.loss_history.append(loss)
    plt.plot(ctx.loss_history)
    plt.ylabel('Loss')

def lay1_overX(ctx, x, netOut):
    x1_yMany(x, netOut, xLabel='Input Range', yLabel='ANN Layer 1', title='ANN Layer 1 over X')
def lay2_overX(ctx, x, netOut):
    x1_yMany(x, netOut, xLabel='Input Range', yLabel='ANN Layer 2', title='ANN Layer 2 over X')
def netOut_overX(ctx, x, netOut):
    x1_yMany(x, netOut, xLabel='Input Range', yLabel='ANN Output', title='ANN Output over X')
#----</CTX functions>----

