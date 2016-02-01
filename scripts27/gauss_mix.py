import tensorflow as tf
import numpy as np

#shape range matrices for calculations
def shape_range_for_elementwise(r, rank=1):
    #r  the range matrix
    #    [ [rb, rt]
    #      [rb, rt] ]
    #s sample size of ins or outs pairs
    r = tf.transpose(r) #group rbot and rtop
    rb, rt = tf.split(0, 2, r) #split rbot and rtop
    
    addRank = rank - 1
    
    for d in range(addRank):
        tf.expand_dims(rb, 1)
        tf.expand_dims (rt, 1)
    
    return rb, rt

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

fullOut_ = tf.placeholder("float", shape=[None, outCount])

#data ranges for elementwise conversion
#[ [rb, rt]
#  [rb, rt] ]
inRange = tf.placeholder("float", shape=[inCount,2]) #2 columns, rtop, rbot
outRange = tf.placeholder("float", shape=[outCount,2])

#two hidden layers
hSize = 5

w1 = tf.Variable(tf.zeros([inCount, hSize]))
b1 = tf.Variable(tf.zeros([hSize]))

w2 = tf.Variable(tf.zeros([hSize, hSize]))
b2 = tf.Variable(tf.zeros([hSize]))

w3 = tf.Variable(tf.zeros([hSize, outCount]))
b3 = tf.Variable(tf.zeros([outCount]))

#get output constants distribution, but mix and make constant
i = np.arange(0, outCount)
np.random.shuffle(i) #in place

#get indices for gauss constants
mixIndex = tf.constant(i[0:g])
varIndex = tf.constant(i[g:g+t])
meanIndex = i[g+t:outCount]
meanIndex = tf.constant(meanIndex.reshape((g,t)))  #shape for later raw gather

sess.run(tf.initialize_all_variables())

# get sample size and convert range matrices for massaging
sCount = tf.gather( tf.shape(fullIn), 0) #NOTE this is a tensor woth one int32 value inside
eRInB, eRInT = shape_range_for_elementwise(inRange, rank=2)
eROutB, eROutT= shape_range_for_elementwise(outRange, rank=3)

# massage inputs to tanh range [-1,1]
fullIn = (2*(fullIn-eRInB)/(eRInT-eRInB)) - 1

#run ANN
lay1 = tf.tanh( tf.matmul(fullIn, w1) + b1 )
lay2 = tf.tanh( tf.matmul(lay1, w2) + b2 )
fullOut = tf.tanh( tf.matmul(lay2, w3) + b3 )

#TEMP until i figure out how to .gather over a dimension (the sample dimension)
#id love to not do this tileShape thing, but sCount is a 0 dim tensor, and cant be paired with a python int scalar
tileShape = tf.concat(0, [tf.convert_to_tensor([1]),tf.expand_dims(sCount,0)])
mixIndex = tf.tile(mixIndex,tileShape)
varIndex =tf.tile(varIndex,tileShape)
meanIndex = tf.tile(meanIndex,tileShape)

#split up fullOut into constants for gaussian mixture
mixRaw = tf.gather(fullOut, mixIndex)
varRaw = tf.gather(fullOut, varIndex)
meanRaw = tf.gather(fullOut, meanIndex)

#massaging functions, prep ANN outputs for use in gaussian mixture, see notes for explanation

#mixing coefficient
mixExp = tf.exp(mixRaw) #e^ each element of mixRaw
mixSum = tf.reduce_sum(mixExp, 1, keep_dims=True) #reduce each sample to scalar, total of all, keep dims for broadcasting over samples
mix = tf.div(mixExp, mixSum) #divide each mixing coefficient by the total.

#variances
var = tf.div(tf.exp(tf.mul(varRaw, 4)), 5) #keep variance positive, and relevant.

#mean
#expand to output range
mean = ((.5 + (meanRaw/2)) * (eROutT - eROutB)) + eROutB

#TODO create loss function...hopefully they have a builtin...at least for max likelihood

