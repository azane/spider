import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

#environmental inputs
envSize = 2
#control feature inputs
conSize = 1

fullIn = tf.placeholder("float", shape=[None, (envSize + conSize)])

#gaussianComponents
g = 5
#targetDims
t = 1

fullOut_ = tf.placeholder("float", shape [None, (g + t + (g*t))])

#two hidden layers
hSize = 5

w1 = tf.variable(tf.zeros([fullIn, hSize]))
b1 = tf.variable(tf.zeros([hSize])

w2 = tf.variable(tf.zeroes([hSize, hSize])
b2 = tf.variable(tf.zeroes([hSize])

w3 = tf.variable(tf.zeroes([hSize, fullOut])
b3 = tf.variable(tf.zeroes([fullOut])

#FIXME do i need to add activation functions to this?

sess.run(tf.initialize_all_variables())

fullOut = tf.sigmoid(tf.multmat(fullIn )
