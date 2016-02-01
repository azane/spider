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
envSize = 3
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
outRange = tf.placeholder("float", shape=[t,2])

#two hidden layers
hSize = 5

w1 = tf.Variable(tf.zeros([inCount, hSize]))
b1 = tf.Variable(tf.zeros([hSize]))

w2 = tf.Variable(tf.zeros([hSize, hSize]))
b2 = tf.Variable(tf.zeros([hSize]))

w3 = tf.Variable(tf.zeros([hSize, outCount]))
b3 = tf.Variable(tf.zeros([outCount]))

def _old():
	#get output constants distribution, but mix and make constant
	#FIXME this was to prevent too much weight sharing...but i can figure outhow to [: [1,4,5]] index in tensorflow! so drop it for contiguousness.
	#i = np.arange(0, outCount)
	#np.random.shuffle(i) #in place

	#get indices for gauss constants
	#mixIndex = tf.constant(i[0:g])
	#np.expand_dims(mixIndex, axis=0) #add a dimension for gather to broadcast over samples.
	#varIndex = tf.constant(i[g:g+t])
	#np.expand_dims(varIndex, axis=0)
	#meanIndex = tf.constant(i[g+t:outCount])
	#meanIndex.reshape((g,t)) ultimately, this needs to be in this shape.
	#np.expand_dims(meanIndex, axis=0)
	pass

sess.run(tf.initialize_all_variables())

# get and convert range matrices for massaging
eRInB, eRInT = shape_range_for_elementwise(inRange, rank=2)
eROutB, eROutT= shape_range_for_elementwise(outRange, rank=3)

# massage inputs to tanh range [-1,1]
fullIn = (2*(fullIn-eRInB)/(eRInT-eRInB)) - 1

#run ANN
lay1 = tf.tanh( tf.matmul(fullIn, w1) + b1 )
lay2 = tf.tanh( tf.matmul(lay1, w2) + b2 )
netOut = tf.tanh( tf.matmul(lay2, w3) + b3 )


def _old():
	#TEMP the numpy version of the operations i want are below. but i can't find a good way to do it in tensorflow.
	#netOut[:, mixIndex]
	#varRaw[:, varIndex]
	#meanRaw[:, meanIndex].reshape((g,t))

	#split up fullOut into constants for gaussian mixture, gather over rows
	#mixRaw = tf.gather(netOut, mixIndex)
	#varRaw = tf.gather(netOut, varIndex)
	#meanRaw = tf.gather(netOut, meanIndex)
	pass

#parse and shape netOut into coefficients
mixRaw = tf.slice(netOut, begin=[0,0], size=[-1,g])
varRaw = tf.slice(netOut, begin=[0,g], size=[-1,t])
meanRaw = tf.reshape(tf.slice(netOut, begin=[0,g+t], size=[-1,-1]), shape=[-1,g,t])


#massaging functions, prep ANN outputs for use in gaussian mixture, see notes for explanation

#mixing coefficient
mixExp = tf.exp(mixRaw) #e^ each element of mixRaw
mixSum = tf.reduce_sum(mixExp, 1, keep_dims=True) #reduce each sample to scalar, total of all, keep dims for broadcasting over samples
mix = tf.div(mixExp, mixSum) #divide each mixing coefficient by the total.

#variances
var = tf.exp(tf.mul(varRaw, 2)) #keep variance positive, and relevant. this scales from tanh output (-1,1)

#mean
#expand to output range
mean = ((.5 + (meanRaw/2)) * (eROutT - eROutB)) + eROutB #this scales from relevant tanh output (-1,1)


#TEMP testing
dataRaw = np.genfromtxt('data/full_with_n.csv', delimiter=',')
f_fullIn = dataRaw[:, [0,1,2,3]]
f_fullOut_ = dataRaw[:,[4]]

f_inRange = np.array(
                        [
                            [f_fullIn[:,0].min(), f_fullIn[:,0].max()],
                            [f_fullIn[:,1].min(), f_fullIn[:,1].max()],
                            [f_fullIn[:,2].min(), f_fullIn[:,2].max()],
                            [f_fullIn[:,3].min(), f_fullIn[:,3].max()]
                        ]
                    )

f_outRange = np.array(
                        [
                            [f_fullOut_[:,0].min(), f_fullOut_[:,0].max()]
                        ]
                    )


finalout = sess.run([mix, var, mean, netOut], feed_dict={
                                                                fullIn:f_fullIn,
                                                                inRange:f_inRange,
                                                                outRange:f_outRange
                                                            })
print "actual (mix, var, mean):"
print 'mix shape: ' + str(finalout[0].shape)
print finalout[0]
print 'var shape: ' + str(finalout[1].shape)
print finalout[1]
print 'mean shape: ' + str(finalout[2].shape)
print finalout[2]



print "netOut:"
print finalout[3]
print "netOut shape: " + str(finalout[3].shape)
print

#TODO create loss function...hopefully they have a builtin...at least for max likelihood
