mport tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

#environmental inputs
envSize = 2
#control feature inputs
conSize = 1

#FIXME inputs should be pre-converted to a range suitable for a sigmoid before processing. 
fullIn = tf.placeholder("float", shape=[None, (envSize + conSize)])

#gaussianComponents
g = 5
#targetDims
t = 1

outCount = g + t + (g*t)

fullOut_ = tf.placeholder("float", shape [None, outCount])

#two hidden layers
hSize = 5

w1 = tf.variable(tf.zeros([fullIn, hSize]))
b1 = tf.variable(tf.zeros([hSize])

w2 = tf.variable(tf.zeroes([hSize, hSize])
b2 = tf.variable(tf.zeroes([hSize])

w3 = tf.variable(tf.zeroes([hSize, fullOut])
b3 = tf.variable(tf.zeroes([fullOut])

#get output constants distribution, but mix and make constant
i = np.arange(0, outCount)
np.random.shuffle(i) #in place

#FIXME may need to convert these to integer tensors?
mixIndex = tf.constant( tf.convert_to_tensor( i[ 0:g-1] ) )
varIndex = tf.constant( tf.convert_to_tensor( i[ g:g+t-1] ) )
meanIndex = tf.convert_to_tensor( i[ g+t:outCount-1] )
meanIndex = tf.constant( tf.reshape(meanIndex, [g,t]) ) #shape for later raw gather

sess.run(tf.initialize_all_variables())

#FIXME may need to transpose some of these.
lay1 = tf.tanh( tf.matmul(fullIn, w1) + b1 )
lay2 = tf.tanh( tf.matmul(lay1, w2) + b2 )
fullOut = tf.tanh( tf.matmul(lay2, w3) + b3 )

#split up fullOut into constants for gaussian mixture
mixRaw = tf.gather(fullOut, mixIndex)
varRaw = tf.gather(fullOut, varIndex)
meanRaw = tf.gather(fullOut, meanIndex)

#massaging functions, prep ANN outputs for use in gaussian mixture, see notes for explanation

#mixing coefficient
mixExp = tf.exp(mixRaw) #e^ each element of mixRaw
mixSum = tf.reduceSum(mixExp) #reduce to scalar, total of all
mix = tf.div(mixExp, mixSum) #divide each mixing coefficient by the total.

#variances
var = tf.div(tf.exp(tf.mul(varRaw, 4)), 5) #keep variance positive, and relevant.

#mean
mean = #TODO see notes for implementation



