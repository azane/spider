import tensorflow as tf
import numpy as np

#shape range matrices for calculations
def shape_range_for_elementwise(r, s):
    #r  the range matrix
    #    [ [rb, rt]
    #      [rb, rt] ]
    #s sample size of ins or outs pairs
    r = tf.transpose(r) #group rbot and rtop
    rb, rt = tf.split(0, 2, r) #split rbot and rtop
    
    rt = tf.tile(rt, s)
    rb = tf.tile(rb, s) #tile up to size
    
    rt = tf.reshape(rt, [s, -1])
    rb = tf.reshape(rb, [s, -1])
    
    return rt, rb

sess = tf.InteractiveSession()

#environmental inputs
envSize = 2
#control feature inputs
conSize = 1

inCount = envSize + conSize

fullIn = tf.placeholder("float", shape=[None, inCount]) #None allows a variable amount of rows, samples in this case


#gaussianComponents
g = 5
#targetDims
t = 1

outCount = g + t + (g*t)

fullOut_ = tf.placeholder("float", shape[None, outCount])

#data ranges for elementwise conversion
#[ [rb, rt]
#  [rb, rt] ]
inRange = tf.placeholder("float", shape[inCount,2]) #2 columns, rtop, rbot
outRange = tf.placeholder("float", shape[outCount,2])

#two hidden layers
hSize = 5

w1 = tf.variable(tf.zeros([inCount, hSize]))
b1 = tf.variable(tf.zeros([hSize]))

w2 = tf.variable(tf.zeroes([hSize, hSize]))
b2 = tf.variable(tf.zeroes([hSize]))

w3 = tf.variable(tf.zeroes([hSize, outCount]))
b3 = tf.variable(tf.zeroes([outCount]))

#get output constants distribution, but mix and make constant
i = np.arange(0, outCount)
np.random.shuffle(i) #in place

#get indices for gauss constants
mixIndex = tf.constant( tf.convert_to_tensor( i[ 0:g-1] ) )
varIndex = tf.constant( tf.convert_to_tensor( i[ g:g+t-1] ) )
meanIndex = tf.convert_to_tensor( i[ g+t:outCount-1] )
meanIndex = tf.constant( tf.reshape(meanIndex, [g,t]) ) #shape for later raw gather

sess.run(tf.initialize_all_variables())

# get sample size and convert range matrices for massaging
sCount = tf.gather( tf.shape(fullIn), 0)
ewiseRangeIn = shape_range_for_elementwise(inRange, sCount)
ewiseRangeOut = shape_range_for_elementwise(outRange, sCount)


#run ANN
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
