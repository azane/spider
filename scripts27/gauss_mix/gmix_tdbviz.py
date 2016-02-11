"""This is a pile of tdb visualization functions that can handle data of arbitrary dimensions (input/output data).
"""

import matplotlib.pyplot as plt
import numpy as np
import gmix_sample_mixture as smpl

#----<helper functions>----
def watch(ctx, y, yLabel='', title=''):
    if not hasattr(ctx, 'history'):
        ctx.history = []
    
    ctx.history.append(y)
    plt.plot(ctx.history)
    
    plt.xlabel('Iterations')
    plt.ylabel(yLabel)
    plt.title(title)

def subs_over1X(x, w, xLabel='Input Range', yLabel='', title=''):
    #stolen, with some mods, from tdb viz example.
    
    x = np.squeeze(np.copy(x))
    assert x.ndim == 1, "This function requires that x can be squeezed to 1 dimension."
    w = np.copy(w)
    
    x, w = sort_by_x1_yMany(x, w)
    
    n = int(np.ceil(np.sqrt(w.shape[2]))) # force square 
    f, axes = plt.subplots(n,n,sharex=True,sharey=True)
    f.suptitle(title)
    
    for i in range(w.shape[2]): # for each target
        r,c=i//n,i%n
        axes[r,c].plot(x, np.squeeze(w[:,:,i]))
        axes[r,c].set_title("[:,:," + str(i) + "]")
        axes[r,c].get_xaxis().set_visible(False)
        axes[r,c].get_yaxis().set_visible(False)
#----</helper functions>----


#----<CTX functions>----
def weights1(ctx, x, w):
    subs_over1X(x, w, yLabel='dE/dW', title='Layer 1 Weight Error Over X')
def weights2(ctx, x, w):
    subs_over1X(x, w, yLabel='dE/dW', title='Layer 2 Weight Error Over X')
def weights3(ctx, x, w):
    subs_over1X(x, w, yLabel='dE/dW', title='Layer 3 Weight Error Over X')

#Outputs
def mixing_coefficients(ctx, x, g):
    x1_yMany(x, g, xLabel='Input Range', yLabel='Relevance', title='Mixing Coefficients over X')

def means(ctx, x, u):
    help_means(x, u)
    #subs_over1X(x, u, title='Means')

def variances(ctx, x, v):
    x1_yMany(x, v, xLabel='Input Range', yLabel='Standard Deviation', title='Standard Deviation Over X')

#Error over X
def calc_grad_v(ctx, x, v):
    x1_yMany(x, v, xLabel='Input Range', yLabel='Error', title='Non TF calculated STD Error Over X')

def calc_grad_m(ctx, x, m):
    x1_yMany(x, m, xLabel='Input Range', yLabel='Error', title='Non TF calculated Mixing Coefficient Error Over X')

def calc_grad_u(ctx, x, u):
    help_means(x, u, yLabel='Error', title='Non TF calculated Mean Error Over X')
    #subs_over1X(x, u, title='Non TF calculated Mean Error Over X')


def tf_grad_v(ctx, x, v):
    x1_yMany(x, v, xLabel='Input Range', yLabel='Error', title='TF STD Error Over X')

def tf_grad_m(ctx, x, m):
    x1_yMany(x, m, xLabel='Input Range', yLabel='Error', title='TF Mixing Coefficient Error Over X')
    #plt.ylim(-2,1)

def tf_grad_u(ctx, x, u):
    help_means(x, u, yLabel='Error', title='TF Mean Error Over X')
    #subs_over1X(x, u, title='TF Mean Error Over X')

def tf_grad_netOut(ctx, x, netOut):
    x1_yMany(x, netOut, xLabel='Input Range', yLabel='Net Out Gradients', title='TF Net Out Gradients')


#Full sample of current mixture model
def sample(ctx, x, m, v, u):
    x, y = smpl.sample_mixture(x, m, v, u)
    plt.scatter(x, y)
    plt.title('Sampling of GMM')

#training data
def training_data(ctx, x, t):
    plt.scatter(x, t)
    plt.title('Training Data')

def watch_loss(ctx, loss):
    watch(ctx, loss, yLabel='Loss', title='Loss')

def watch_grad_m(ctx, grad_m):
    watch(ctx, np.mean(np.absolute(grad_m), axis=0), title='Mixing Coefficient Wrongness')
    
def watch_grad_v(ctx, grad_v):
    watch(ctx, np.mean(np.absolute(grad_v), axis=0), title='Standard Deviation Wrongness')

def watch_grad_u(ctx, grad_u):
    watch(ctx, np.mean(np.absolute(grad_u), axis=0), title='Standard Deviation Wrongness')
    

def lay1_overX(ctx, x, netOut):
    x1_yMany(x, netOut, xLabel='Input Range', yLabel='ANN Layer 1', title='ANN Layer 1 over X')
def lay2_overX(ctx, x, netOut):
    x1_yMany(x, netOut, xLabel='Input Range', yLabel='ANN Layer 2', title='ANN Layer 2 over X')
def netOut_overX(ctx, x, netOut):
    x1_yMany(x, netOut, xLabel='Input Range', yLabel='ANN Output', title='ANN Output over X')

def report_net(ctx, w1, b1, w2, b2, w3, b3, hSize):
    
    """
    This functions creates a pile of subplots that sensibly displays the iteration series of
     the weights and biases of a network with 2 hidden layers of variable size.
    """
    
    return #TEMP
    
    #---<Series>---
    if not hasattr(ctx, 'net_history'):
        ctx.net_history = dict(
            w1=[],
            w2=[],
            w3=[],
            b1=[],
            b2=[],
            b3=[]
        )
    ctx.net_history['w1'].append(w1)
    ctx.net_history['w2'].append(w2)
    ctx.net_history['w3'].append(w3)
    ctx.net_history['b1'].append(b1)
    ctx.net_history['b2'].append(b2)
    ctx.net_history['b3'].append(b3)
    #---</Series>---
    
    #---<Display>---
    if not hasattr(ctx, 'net_info'):
        cols = (hSize*hSize) + (hSize-1) #for hSize groups of hSize, with one space in between each.
        if cols%2 == 0:
            mid = cols/2
            #if even, leave two spaces in the middle, and add one to cols for the extra space.
            dblMid = True
            cols += 1
        else:
            mid = (cols/2) + 1  # set mid to actual middle
            dblMid = False
        
        rows = 7
        
        ctx.net_info = dict(
            cols=cols,
            rows=rows,
            mid=mid,
            dblMid=dblMid,
            hSize=hSize
        )
    
    #TODO
    #for r in range(ctx.net_info['hSize']): #  iterate layer size
        
    
    
    """in
    
    w*5
    b*5 and activations
    
    w*5
    b*5 and activations
    
    w*5
    b*5 and activations
    """
    
    #---</Display>---
    
#----</CTX functions>----

#----<Op Functions>----

#----<Op Functions>----
