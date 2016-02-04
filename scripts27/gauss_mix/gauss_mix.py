import tensorflow as tf
import numpy as np
import sys

SUMMARY_DIRECTORY = "/Users/azane/GitRepo/spider/data/gauss_mix_logs" #directory in which summary files are stored
XMVU_PATH = "/Users/azane/GitRepo/spider/data/xmvu.npz" #file to which the xmvu data, for sampling is written after training.




"""brainstorming
with the visualization implementation, we're going to need to clean this file up, and put it in a class probably.

one file with visualization functions

one file with the gmix class. this will need to make new graphs/sessions on __init__,
    and it will need a flag to tell if and what summaries/reporting structure to use

one file that functions as the command line face, i.e. remove the main clause, and put it in another file that handles iterations and stuff, but in functions.
    perhaps this should function as the api face of the gmix model too?


CRITICAL PATH:
1. move the main section to another file and seperate it into functions, callable from the main bit, but useable from elsewhere.
2. do some cleaning on the gmix file, and organize it into a class. implement the graph/session inits and report flags.
3. write the visualization file and add tensorflow debugger functionality to the gmix file.
"""

#TODO throughout this project, make 't' be the variable for training targets, i.e. ideal outputs
#       and 'y' the variable for actual outputs.

"""
command line calling structure
:python thisfile.py <str:train.npz> <str:test.npz>
   where train.npz and test.npz should have two arrays, x and t.
   and x.shape == (s,inDims)
   and t.shape == (s,outDims)
train and test should obviously be different sets of data from the same source.

"""

#TODO make this a class that inits graphs/sessions for each new instance. this will be required to have lots of jasons running their own graphs.
#       or lots of jasons calling to the same server to run their data and return their weight matrices.
#     we'll also need to control which instances get summaries written for them. just with a flag probs.

#TODO make the variable naming more consistent throughout functions. it's confusing...slash, that may come with making it a class.

#TODO where possible, convert range conversions to this process:
#   data -= data.min() #make 0 base
#   data /= data.max() #divide by maximum
#   data *= 2 #multiply 0-1 range for [0,2] range
#   data -= 1 #subtract one to shift for [-1,1] range.
#   #FIXME actually, this may not work very well, because a sample batch may not capture the range the spider decides it wants to search over. : /
#   #       but i'm still overcomplicating the math. this is better.
#   #FIXME also, to expand the range back out, it needs to remember it...so maybe my current method is so far gone?

def weight_variable(shape):
    """This returns a randomized weight matrix"""
    initial = tf.truncated_normal(shape, stddev=0.5)
    return tf.Variable(initial)

def bias_variable(shape):
    """This returns a nonzero bias matrix"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def shape_range_for_elementwise(r, rank=1):
    """This shapes range matrices for easier calculations."""
    
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

def gmix_model(inCount, t, g=5, hSize=15):
    """ This function builds the tensorflow graph and returns a dict for reference.
            This does not build a loss function.
                It only builds the placeholders, ANN, and shapes the ANN outputs into something useable by a loss function or a sampler.
        
        inCount is the number of x inputs, i.e. the number of input dimensions (not array dimensions!)
        t is the number of ultimate target dimensions, this differs from the output of the ANN
        g is the number of gaussian components to employ
        hSize is the size of the hidden layers of the ANN
    """
    
    #TODO make g and hSize defaults a function of data complexity? or maybe have them be learned as well.
    #TODO FIXME make naming in this function consistent with the rest of the project.
    

    #-----<Placeholder Construction>-----
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
        hSize = 15

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
        #varScale = tf.Variable(2.0) #TODO is this a good idea?
        var = tf.exp(tf.mul(varRaw, 2.0)) #keep variance positive, and relevant. this scales from tanh output (-1,1)

    with tf.name_scope('massage_mean') as scope:
        #mean
        #expand to output range
        mean = ((.5 + (meanRaw/2)) * (eROutT - eROutB)) + eROutB #this scales from relevant tanh output (-1,1)

    #summary ops
    mix_hist = tf.histogram_summary("Mixing Coefficients", mix)
    var_hist = tf.histogram_summary("Variances", var)
    mean_hist = tf.histogram_summary("Means", mean)
    #-----</Massage for Gaussian Mixture>-----
    
    #-----<Build Reference Dict>-----
    #   this section builds and returns a dictionary that can be used from outside this function to run tensorflow output and to fill placeholders.
    rDict = dict(
        x=fullIn,
        t=fullOut_,
        u=mean, #as in 'mew'
        v=var,
        m=mix,
        inRange=inRange,
        outRange=outRange,
        netIn=netIn,
        mixRaw=mixRaw,
        varRaw=varRaw
    )
    
    return rDict
    #-----</Build Reference Dict>-----


def gmix_nll(m, v, u, t):
    """
        This defines the nll loss function for a gaussian mixture model.
        
        (s is the sample size, g is the number of gaussian components, t is the number of target components)
        where:
           m is the mixing coefficent array of shape [s,g]
           v is the variance array of shape [s,t]
           u is the mean (mew) array of shape [s,g,t]
           t is the target array of shape [s,t]
    
    """
    
    #-----<Assertions>----- #FIXME these have to be tf.eval() 'd to work. otherwise they stand for unique tensor operations.
    #print str(tf.shape(m))
    #assert tf.shape(m)[0] == tf.shape(v)[0] and tf.shape(u)[0] == tf.shape(t)[0] and tf.shape(u)[0] == tf.shape(m)[0] #assert that s is equal across.
    #assert tf.shape(m)[1] == tf.shape(u)[1] #assert that g is equal across
    #assert tf.shape(v)[1] == tf.shape(u)[2] and tf.shape(v)[1] == tf.shape(t)[1] #assert that t is equal across
    #-----</Assertions>-----
    
    
    with tf.name_scope('mixture_nll') as scope:

        #prep terms of variance
        #add a dimension of 1 to be broadcast over the 'g' dimension in other tensors.
        v = tf.expand_dims(v, 1) #now tf.shape(v) == (s, 1, t)
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

def infer_space(x, t):
    """This function infers the dimensions of the input and output spaces given the input and target arrays.
        x.shape == (s, inDims)
        t.shape == (s, outDims)
    """
    
    inDims = x.shape[1] #inCount
    outDims = t.shape[1] #t
    
    return inDims, outDims

def verify_xt(x, t):
    """This function merely serves to verify x and t, and is called from a number of possible xt entry points."""
    #---<Verify>---
    assert x.shape[0] == t.shape[0], \
        "input and output arrays must have the same sample size. i.e. x.shape[0] == t.shape[0]. x.shape: %s, t.shape: %s! " % (str(x.shape), str(t.shape))
    assert x.ndim == 2, "The x array must have shape (sampleSize, inputDimensions), not " % (str(x.shape))
    assert t.ndim == 2, "The t array must have shape (sampleSize, inputDimensions), not " % (str(t.shape))
    #---</Verify>---
    
    #---<Warn>---
    if x.shape[0] <= x.shape[1]:
        #TODO use the warnings module to raise a warning
        print "WARNING: There are fewer samples than input dimensions, x is probably transposed incorectly. x.shape == %s" % (str(x.shape))
    if t.shape[0] <= t.shape[1]:
        #TODO use the warnings module to raise a warning
        print "WARNING: There are fewer samples than output dimensions, t is probably transposed incorectly. t.shape == %s" % (str(t.shape))
    #---</Warn>---

def get_xt_from_npz(npz_path):
    """This function reads an npz file, verifies xt, and returns the unpacked arrays."""
    
    with np.load(npz_path, allow_pickle=False) as xt:
        x = xt['x']
        try:
            t = xt['t']
        except KeyError:
            t = xt['y'] #try under y.
    
    verify_xt(x, t)
    
    return x, t
    
def infer_ranges(x, t):
    """This function infers the data ranges from the data. It returns two arrays like this:
            [ [rb, rt]
              [rb, rt] ]
        where 'rb' and 'rt' are the bottom and top, respectively, of the range for the dimension corresponding to their index.
    """
    #----<Infer Ranges>----
    #can this be done without loops? it doesn't happen often, so it's probs nbd. and the loops are small.
    inDims, outDims = infer_space(x, t)
    
    inRange = []
    for d in range(inDims): #-1 for count vs. index
        inRange.append([x[:,d].min(), x[:,d].max()]) #get the min/max over all samples for each input dimension.
    
    outRange = []
    for d in range(outDims):
        outRange.append([t[:,d].min(), t[:,d].max()])
    #----</Infer Ranges>----
    
    return np.array(inRange), np.array(outRange) #convert to np.array and return

def gmix_training_model(inDims, outDims):
    """This function pieces things together into a trainable model.
    
    - builds the mixture model that takes inputs and returns, means, variances, and mixture coefficients.
    - uses those values and builds the loss model.
    - builds a training step
    - initializes the session and variables
    - merges and preps summary writing
    
    """
    
    #ins -> mvu
    with tf.name_scope('model'):
        modelDict = gmix_model(inDims, outDims, g=5, hSize=20) #TODO offshore g and hSize determination
    
    #loss
    #collect mvut placeholders from model for loss function.
    m, v, u, t = modelDict['m'], modelDict['v'], modelDict['u'], modelDict['t']
    with tf.name_scope('loss') as scope:
        loss = gmix_nll(m, v, u, t)
    
    #train
    with tf.name_scope('train_step') as scope:
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        gradients = optimizer.compute_gradients(loss)
        train_step = optimizer.minimize(loss)
    
    #init session
    sess = tf.Session()

    #init summary writer
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(SUMMARY_DIRECTORY, sess.graph_def)

    #init vars
    sess.run(tf.initialize_all_variables())
    
    #append new elements to model dict
    modelDict['sess']=sess
    modelDict['summaries']=merged
    modelDict['summaryWriter']=writer
    modelDict['train_step']=train_step
    modelDict['loss']=loss
    modelDict['gradients']=gradients
    modelDict['train_step']=train_step
    
    return modelDict

def sample_batch(x, t, size):
    i = np.random.random_integers(0, high=x.shape[0]-1, size=size)
    return x[i], t[i]

#----------------------------------------------------------------<MAIN>----------------------------------------------------------------
if __name__ == "__main__":
    #TODO TODO split up this main section. make a function that can be called just with x and t array values
    #       cz the only reason to really use the command line is to read from a file...so that's really the only thing we need in the main section.
    
    #----<Read Data>----
    #t for test, s for training sample
    s_x, s_t = get_xt_from_npz(sys.argv[1])
    t_x, t_t = get_xt_from_npz(sys.argv[2])
    #----</Read Data>----
    
    #----<Training Constants>----
    ITERATIONS = range(1000) #iterable
    
    TEST_BATCH_SIZE = 500
    if t_x.shape[0] < TEST_BATCH_SIZE:
        TEST_BATCH_SIZE = t_x.shape[0]
        
    TRAIN_BATCH_SIZE = 1000
    if s_x.shape[0] < TRAIN_BATCH_SIZE:
        TRAIN_BATCH_SIZE = s_x.shape[0]
    #----</Training Constants>----
    
    #----<Training Setup>----
    inDims, outDims = infer_space(s_x, s_t) #get info to build model
    modelDict = gmix_training_model(inDims, outDims) #build model, get dict of tensorflow placeholders, variables, outputs, etc.
    
    inRange, outRange = infer_ranges(s_x, s_t) #infer ranges for feed dict
    feed_dict = {
                    modelDict['x']:None,
                    modelDict['t']:None,
                    modelDict['inRange']:inRange,
                    modelDict['outRange']:outRange
                } #build feed dict, but leave x and t empty, as they will be updated in the training loops.
    #----</Training Setup>----
    
    #----<Training Loop>----
    for i in ITERATIONS:
        
        if i % 10 == 0: #run reports every 10 iterations.
            feed_dict[modelDict['x']], feed_dict[modelDict['t']] = sample_batch(t_x, t_t, TEST_BATCH_SIZE) #update feed_dict with test batch
            
            result = modelDict['sess'].run([
                                            modelDict['summaries'],
                                            modelDict['loss']
                                        ], feed_dict=feed_dict) #run model with test batch
            
            modelDict['summaryWriter'].add_summary(result[0], i) #write to summary
            print("Loss at step %s: %s" % (i, result[1])) #print loss
            
            print '-------------------------------'
            
        else:
            #update feed_dict with training batch
            feed_dict[modelDict['x']], feed_dict[modelDict['t']] = sample_batch(s_x, s_t, TRAIN_BATCH_SIZE)
            modelDict['sess'].run([
                                    modelDict['train_step']
                                ], feed_dict=feed_dict) #run train_step with training batch
    #----<Training Loop>----
    
    #----<Store for Sampling>----
    feed_dict[modelDict['x']], feed_dict[modelDict['t']] = t_x, t_t #update feed_dict with full test data batch
    #evaluate the trained model at m, v, and u with the test data.
    result = modelDict['sess'].run([
                                    modelDict['m'],
                                    modelDict['v'],
                                    modelDict['u']
                                ],feed_dict=feed_dict)
    np.savez(XMVU_PATH, x=t_x, m=result[0], v=result[1], u=result[2]) #write this to an npz file for later sampling
    #----</Store for Sampling>----
