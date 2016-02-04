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