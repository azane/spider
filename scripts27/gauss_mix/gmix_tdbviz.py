"""This is a pile of tdb visualization functions that can handle data of arbitrary dimensions (input/output data).
"""

import matplotlib.pyplot as plt
import numpy as np
import gmix_sample_mixture as smpl

#----<helper functions>----
def watch1graph(ctx, y, yLabel='', title=''):
    if not hasattr(ctx, 'history'):
        ctx.history = []
    
    ctx.history.append(y)
    plt.plot(ctx.history)
    
    plt.xlabel('Iterations')
    plt.ylabel(yLabel)
    plt.title(title)
def square(data, title='', normalize=True, cmap=plt.cm.gray, padsize=1, padval=0):
    #stolen , with some mods, from tdb viz example file. FIXME not working 20160211
    """
    takes a np.ndarray of shape (n, height, width) or (n, height, width, channels)
    visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    However, this only draws first input channel
    """
    # normalize to 0-1 range
    if normalize:
        data -= data.min()
        data /= data.max()
    n = int(np.ceil(np.sqrt(data.shape[0]))) # force square 
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.matshow(data,cmap=cmap)
    plt.title(title)

def subs_over_iterations(data, title=''):
    #stolen, with some mods, from tdb viz example.
    
    #data must be 3d, the x axis will be the first dimension.
    #the graphs are divided by the last dimension
    
    n = int(np.ceil(np.sqrt(data.shape[2]))) # force square 
    f, axes = plt.subplots(n,n,sharex=True,sharey=True)
    f.suptitle(title)
    
    for i in range(data.shape[2]):
        r,c=i//n,i%n
        axes[r,c].plot(np.squeeze(data[:,:,i])) #squeeze down to 2d, selecting one of the outer dimension
        axes[r,c].set_title("[:,:," + str(i) + "]") #set the title to this index
        axes[r,c].get_xaxis().set_visible(False)
        axes[r,c].get_yaxis().set_visible(False)
#----</helper functions>----


#----<CTX functions>----
def watch_loss(ctx, loss):
    watch1graph(ctx, loss, yLabel='Loss', title='Loss')

def weight_hist(ctx, w):
    plt.hist(w.flatten())

def weight_squares(ctx, w):
    square(w, title='Weights Now')

def watch_weights(ctx, w):
    if not hasattr(ctx, 'history'):
        ctx.history = []
    ctx.history.append(w)
    subs_over_iterations(np.array(ctx.history))

def watch_biases(ctx, b):
    watch1graph(ctx, b, yLabel='Bias', title='Bias Over Iterations')

#----</CTX functions>----

#----<Op Functions>----

#----<Op Functions>----
