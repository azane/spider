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
import gmix_sample_mixture


def x1_g(ctx, x, g, yLabel='Mixing Coefficients'):
    #x.shape == [s,x]
    #g.shape == [s,g]
    #TODO plot an arbitrary number of mixing lines.
    print('g[:,0].shape: ' + str(g[:,0].shape))
    plt.plot(x[:,0], g[:,0])
    plt.xlabel('Input Range')
    plt.ylabel(yLabel)

def x1_t(ctx, x, t):
    pass

def x1_g_t(ctx, x, g, t):
    pass

def watch_loss(ctx, loss):
  if not hasattr(ctx, 'loss_history'):
    ctx.loss_history=[]
  ctx.loss_history.append(loss)
  plt.plot(ctx.loss_history)
  plt.ylabel('Loss')
