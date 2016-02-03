import tensorflow as tf
import numpy as np

##TODO command line calling structure
#python thisfile.py <str:x.csv> <str:t.csv> <int:gaussian components> <int: ANN hidden layer size>
#TODO infer in and outsize by the number of columns in the x/t csv files.
#TODO put all this in a __main__ section so that it can be called from the command line, or as a python function. : )

#shape range matrices for calculations
def shape_range_for_elementwise(r, rank=1):
    #r  the range matrix
    #    [ [rb, rt]
    #      [rb, rt] ]
    r = tf.transpose(r) #group rbot and rtop
    rb, rt = tf.split(0, 2, r) #split rbot and rtop
    
    addRank = rank - 1
    
    for d in range(addRank):
        tf.expand_dims(rb, 1)
        tf.expand_dims (rt, 1)
    
    return rb, rt

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


#-----<Mixture Construction>-----
#environmental inputs
envSize = 4
#control feature inputs
conSize = 0
#targetDims
t = 1
#gaussianComponents
g = 3
#-----</Mixture Construction>-----


#-----<Placeholder Construction>-----
inCount = envSize + conSize #for ANN and, by proxy, gaussian mixture
outCount = g + t + (g*t) #for ANN

fullIn = tf.placeholder("float", shape=[None, inCount], name='x_val') #None allows a variable amount of rows, samples in this case

fullOut_ = tf.placeholder("float", shape=[None, t], name='t_val') #for gaussian mixture


#data ranges for elementwise conversion
#[ [rb, rt]
#  [rb, rt] ]
inRange = tf.placeholder("float", shape=[inCount,2], name="inRange") #2 columns, rtop, rbot
outRange = tf.placeholder("float", shape=[t,2], name="outRange")

# get and convert range matrices for massaging
eRInB, eRInT = shape_range_for_elementwise(inRange, rank=2)
eROutB, eROutT= shape_range_for_elementwise(outRange, rank=3)
    
#summary ops
x_hist = tf.histogram_summary("x", fullIn)
t_hist = tf.histogram_summary("t", fullOut_)
#-----</Placeholder Construction>-----


#-----<ANN Construction>-----
with tf.name_scope('ANN_const') as scope:
    #two hidden layers of size:
    hSize = 30

    w1 = weight_variable([inCount, hSize])
    b1 = bias_variable([hSize])

    w2 = weight_variable([hSize, hSize])
    b2 = bias_variable([hSize])

    w3 = weight_variable([hSize, outCount])
    b3 = bias_variable([outCount])

#summary ops
w1_hist = tf.histogram_summary("w1", w1)
b1_hist = tf.histogram_summary("b1", b1)

w2_hist = tf.histogram_summary("w2", w2)
b2_hist = tf.histogram_summary("b2", b2)

w3_hist = tf.histogram_summary("w3", w3)
b3_hist = tf.histogram_summary("b3", b3)
#-----</ANN Construction>-----

#-----<ANN Execution>-----
with tf.name_scope('ANN_exec') as scope:
    # massage inputs to tanh range [-1,1]
    netIn = (2*(fullIn-eRInB)/(eRInT-eRInB)) - 1

    #run ANN
    lay1 = tf.tanh( tf.matmul(netIn, w1) + b1 )
    lay2 = tf.tanh( tf.matmul(lay1, w2) + b2 )
    netOut = tf.tanh( tf.matmul(lay2, w3) + b3 )

    #parse and shape netOut activations into coefficient arrays.
    mixRaw = tf.slice(netOut, begin=[0,0], size=[-1,g]) #.shape == [s,g]
    varRaw = tf.slice(netOut, begin=[0,g], size=[-1,t]) #.shape == [s,t]
    meanRaw = tf.reshape(tf.slice(netOut, begin=[0,g+t], size=[-1,-1]), shape=[-1,g,t]) #.shape == [s,g,t]
#-----</ANN Execution>-----


#-----<Massage for Gaussian Mixture>-----
#massaging functions, prep ANN outputs for use in gaussian mixture, see notes for explanation

with tf.name_scope('massage_mix') as scope:
    #mixing coefficient
    mixExp = tf.exp(mixRaw) #e^ each element of mixRaw
    mixSum = tf.reduce_sum(mixExp, 1, keep_dims=True) #reduce each sample to scalar, total of all, keep dims for broadcasting over samples
    mix = tf.div(mixExp, mixSum) #divide each mixing coefficient by the total.

with tf.name_scope('massage_var') as scope:
    #variances
    #FIXME TODO we may want to make this a function of the range of the target for which this variance will be used.
    var = tf.exp(tf.mul(varRaw, 7)) #keep variance positive, and relevant. this scales from tanh output (-1,1)

with tf.name_scope('massage_mean') as scope:
    #mean
    #expand to output range
    mean = ((.5 + (meanRaw/2)) * (eROutT - eROutB)) + eROutB #this scales from relevant tanh output (-1,1)

#summary ops
mix_hist = tf.histogram_summary("Mixing Coefficients", mix)
var_hist = tf.histogram_summary("Variances", var)
mean_hist = tf.histogram_summary("Means", mean)
#-----</Massage for Gaussian Mixture>-----


def mixture_negative_log_likelihood(m, v, u, t):
    with tf.name_scope('mixture_nll') as scope:
        #(s is the sample size, g is the numbe of gaussian components, t is the number of target components)
        #where:
        #   m is the mixing coefficent array of shape [s,g]
        #   v is the variance array of shape [s,t]
        #   u is the mean (mew) array of shape [s,g,t]
        #   t is the target array of shape [s,t]

        #prep terms with variance inside.
        #add a dimension of 1 to be broadcast over the 'g' dimension in other tensors.
        v = tf.expand_dims(v, 1) #add after dim index 1 for shape [s, 1, t]
        v_norm = 1/(v*tf.sqrt(2*np.pi))
        v_dem = 2*tf.square(v)

        #prep numerator term with corresponding sample target values and net proposed means
        t = tf.expand_dims(t, 1) #add dim at 2nd position for broadcasting over means along dimension g.
        tm_num = tf.square(t-u)

        #employ terms in pre-mixed likelihood function.
        premix = v_norm*(tf.exp(-(tm_num/v_dem)))
        #add dim for broadcasting mixing coefficients over t, the 3rd dimension.
        m = tf.expand_dims(m, 2)

        #mix gaussian components
        likelihood = m*premix

        #sum over the likilihood of each target and gaussian component, don't reduce over samples yet.
        #FIXME i know the gaussian components are supposed to be summed, but i'm not sure about the various targets?
        #FIXME      do i have to mix the targets like the gaussian components?
        tot_likelihood = tf.reduce_sum(likelihood, [1,2]) #sum over g and t dimensions.

        #take natural log of sum, then reduce over samples, then negate for the final negative log likelihood
        nll = -tf.reduce_sum(tf.log(tot_likelihood)) #this reduces along the final dimension, so nll will be a scalar.
    
    #summary ops
    v_norm_hist = tf.histogram_summary("Variance Normalizer", v_norm)
    v_dem_hist = tf.histogram_summary("Variance Denominator", v_dem)
    tm_num_hist = tf.histogram_summary("Target-Mean Numerator", tm_num)
    premix_hist = tf.histogram_summary("Pre-mixed Likelihood", premix)
    likelihood_hist = tf.histogram_summary("Post-mixed Likelihood", likelihood)
    tot_likelihood_hist = tf.scalar_summary("Average Likelihood Across Samples", tf.reduce_mean(tot_likelihood))
    nll_hist = tf.scalar_summary("Negative Log Likelihood", nll)
    
    return nll


#----<Read Data>----
dataRaw = np.genfromtxt('data/full_with_n.csv', delimiter=',')

#t for test, f for feed
t_fullIn = dataRaw[0:dataRaw.shape[0]:2, [0,1,2,3]]
t_fullOut_ = dataRaw[0:dataRaw.shape[0]:2,[4]]

f_fullIn = dataRaw[1:dataRaw.shape[0]:2, [0,1,2,3]]
f_fullOut_ = dataRaw[1:dataRaw.shape[0]:2,[4]]

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
#----</Read Data>----#TODO separate out the data reading. it shouldn't be in this same file.

with tf.name_scope('loss') as scope:
    loss = mixture_negative_log_likelihood(mix, var, mean, fullOut_)

with tf.name_scope('train_step') as scope:
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


#init session
sess = tf.Session()

#init summary writer
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("data/gauss_mix_logs", sess.graph_def)

#init vars
sess.run(tf.initialize_all_variables())


#training loop
for i in range(1000):

    if i % 20 == 0:
        s_i = np.random.random_integers(0, high=t_fullIn.shape[0]-1, size=500)
        f_dict = {
                    fullIn:t_fullIn[s_i],
                    inRange:f_inRange,
                    outRange:f_outRange,
                    fullOut_:t_fullOut_[s_i]
                    
                    #overfitting test:
                    #fullIn:f_fullIn[s_i],
                    #fullOut_:f_fullOut_[s_i]
                }

        result = sess.run([merged, loss, mean], feed_dict=f_dict)
        summary_str = result[0]
        test_loss = result[1]
        writer.add_summary(summary_str, i)
        #writer.flush()
        
        #---<Make more flexible with range>----
        #sampleSize = result[2].shape[0]
        #viewSize = 20
        #manualRange = .0015
        
        #test_means = result[2][0:sampleSize:int(sampleSize/viewSize)]
        #test_targets = np.expand_dims(t_fullOut_[s_i][0:sampleSize:int(sampleSize/viewSize)], 1)
        #test_diff = np.abs(test_targets - test_means)
        #test_best_mean_diff = np.amin(test_diff, axis=1)
        #test_avg_diff = np.mean(test_best_mean_diff)
        #----</MMFWR>----
        
        print("Loss at step %s: %s" % (i, test_loss))
        #<MMFWR>print("Average target distance from best mean as percentage of range at step %s: %s" % (i, test_avg_diff/manualRange))#</MMFWR
        print "--------"

    else:
        s_i = np.random.random_integers(0, high=f_fullIn.shape[0]-1, size=50)
        f_dict = {
                    fullIn:f_fullIn[s_i],
                    inRange:f_inRange,
                    outRange:f_outRange,
                    fullOut_:f_fullOut_[s_i]
                }

        sess.run(train_step, feed_dict=f_dict)
