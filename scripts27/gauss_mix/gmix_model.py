import tensorflow as tf
import numpy as np
import sys
from gmix_sample_mixture import sample_mixture
import gmix_tdbviz as viz
import tdb as tdb

SUMMARY_DIRECTORY = "/Users/azane/GitRepo/spider/data/gauss_mix_logs" #directory in which summary files are stored

#-----<Helper Functions>-----
def get_xt_from_npz(npz_path):
    """Return x and t
        
        This function reads an npz file, verifies xt, and returns the unpacked arrays.
        
        'x' must be under npz['x']
        't' must be under either npz['t'] or npz['y']
    """

    with np.load(npz_path, allow_pickle=False) as xt:
        x = xt['x']
        try:
            t = xt['t']
        except KeyError:
            t = xt['y'] #try under y.

    return x, t
#-----</Helper Functions>-----

"""brainstorming for time series tdb
keep a pile of np.array 1d objects that get data appended to them when the process runs.
feed these arrays, into the debug session, into the placeholders that are graphed by tdb plot ops.
the debug session will then take the data from the placeholders, and return evaluated plot ops, ready for graphing.

we should keep a dictionary of these placeholders, and extend the feed dictionary, automatically defining stuff, when it gets plugged into the debug session.

"""

class GaussianMixtureModel(object):
    def TODOsFIXMEs():
        pass
        
        #TODO make docstring for this class
        #TODO throughout this project, make 't' be the variable for training targets, i.e. ideal outputs
        #       and 'y' the variable for actual outputs.
        
        
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
        
    def __init__(self, x, t, x_test, t_test, inDims=None, outDims=None, inRange=None, outRange=None, numGaussianComponents=5, hiddenLayerSize=15, learningRate=0.01):
        
        self.graph = tf.Graph()
        #set this graph as default for every public function that uses tensorflow.
        with self.graph.as_default():
        
            #-----<Attribute Processing>-----
            #TODO allow '*_test' to be None
            #       in that case, generate a sample test batch of 'test_size' and separate it from x and t used for training.
            self._verify_xt(x, t)
            self._verify_xt(x_test, t_test)
        
            self.x = x
            self.t = t
            self.x_test = x_test
            self.t_test = t_test
        
            if inDims is None:
                self.inDims = self._infer_space(x, t)[0]
            if outDims is None:
                self.outDims = self._infer_space(x, t)[1]
        
            if inRange is None:
                self.inRange = self._infer_ranges(x, t)[0]
            if outRange is None:
                self.outRange = self._infer_ranges(x, t)[1]
        
            self.numGaussianComponents = numGaussianComponents
            self.hiddenLayerSize = hiddenLayerSize
            self.learningRate = learningRate
            #-----</Attribute Processing>-----
        
            #this is a reference dictionary where tensorflow tensor objects are stored.
            #   it's basically employed to keep the class namespace cleaner, cz there can be looots of tensors.
            self.refDict = {}
        
            self._gmix_training_model()
            
            self._tdb_nodes()
        
    def _infer_space(self, x, t):
        """Return inferred in and out dimensions
            
            This function infers the dimensions of the input and output spaces given the input and target arrays.
            x.shape == (s, inDims)
            t.shape == (s, outDims)
        """
        
        inDims = x.shape[1]
        outDims = t.shape[1] #t
        
        return inDims, outDims
        
    def _verify_xt(self, x, t):
        """Verify the shapes of x and t
            
            This function merely serves to verify x and t, and is called from a number of possible xt entry points.
        """
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
    
    def _infer_ranges(self, x, t):
        """Return inferred in and out ranges.
        
        This function infers the data ranges from the data. It returns two arrays like this (for match the input to the range shaping method):
                [ [rb, rt]
                  [rb, rt] ]
            where 'rb' and 'rt' are the bottom and top, respectively, of the range for the dimension corresponding to their index.
        """
        #----<Infer Ranges>----
        #can this be done without loops? it doesn't happen often, so it's probs nbd. and the loops are small.
        
        inRange = []
        for d in range(self.inDims):
            inRange.append([x[:,d].min(), x[:,d].max()]) #get the min/max over all samples for each input dimension.
            
        outRange = []
        for d in range(self.outDims):
            outRange.append([t[:,d].min(), t[:,d].max()])
        #----</Infer Ranges>----
        
        return np.array(inRange, dtype=np.float32), np.array(outRange, dtype=np.float32)  # convert from list
        
    def _weight_variable(self, shape):
        """Return weight matrix
        This initializes weight matrices for the ANN in this model.
        """
        #initialize weights on a truncated normal distribution
        initial = tf.truncated_normal(shape, stddev=0.5)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        """Return bias matrix
        This initializes bias matrices for the ANN in this model.
        """
        #initialize non-zero bias matrix
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _shape_range_for_elementwise(self, r, rank=1):
        """Return reshaped range matrices
        This shapes range matrices for easier calculations.
        """
    
        #r: the range matrix
        #    [ [rb, rt]
        #      [rb, rt] ]
        r = tf.transpose(r)  # group rbot and rtop into rows
        rb, rt = tf.split(0, 2, r)  # split range bottom and range top (rb, rt) into two tensors.
    
        #expand dimensions until the defined rank is reached.
        addRank = rank - 1
        for d in range(addRank):
            tf.expand_dims(rb, 1)
            tf.expand_dims (rt, 1)
    
        return rb, rt

    def _sample_batch(self, x, t, size):
        """Return a randomly sampled batch from x and t of size 'size'
        """
        i = np.random.random_integers(0, high=x.shape[0]-1, size=size)
        return x[i], t[i]
    def _gmix_forward_model(self):
        """Build forward model and add tensors to the dict of tensorflow tensors
    
            This method builds the forward part of the tensorflow graph and returns a dict for reference.
                This does not build a loss function. It only builds the placeholders, ANN,
                    and shapes the ANN outputs into something useable by a loss function or a sampler.
        
            'inDims' is the size of the x vector, i.e. the number of input dimensions (not array dimensions!)
            'outDims' is the size of the t/y vector, i.e. the number of ultimate target dimensions, this differs from the output of the ANN
            'g' is the number of gaussian components to employ
            'hiddenLayerSize' is the size of the hidden layers of the ANN
        """
        
        #TODO make g and hiddenLayerSize defaults a function of data complexity? or maybe have them be learned as well.
        
        #TODO FIXME make naming in this function consistent with the rest of the project.
        #               change 'g' to something that does not imply it's array_like (it's an int). elsewhere, single letter variables are array_like, 
        #                   and used in computation. #update: just use the instance variable, self.numGaussianComponents
        #               change fullIn and fullOut_ names to x and t.
        #               change mix, mean, and var to their single letter names. m, u, and v.
        
        #TODO this was a conversion from an independent function, so just use the instance attributes directly in this method.
        inDims = self.inDims
        outDims = self.outDims
        g = self.numGaussianComponents
        hiddenLayerSize = self.hiddenLayerSize
        
        
        #-----<Placeholder Construction>-----
        outCount = g + outDims + (g*outDims) #for ANN
        
        fullIn = tf.placeholder(tf.float32, shape=[None, inDims], name='x_val') #None allows a variable amount of rows, samples in this case
        
        fullOut_ = tf.placeholder(tf.float32, shape=[None, outDims], name='t_val') #for gaussian mixture
        
        
        #data ranges for elementwise conversion
        #[ [rb, rt]
        #  [rb, rt] ]
        inRange = tf.placeholder("float", shape=[inDims,2], name="inRange") #2 columns, rtop, rbot
        outRange = tf.placeholder("float", shape=[outDims,2], name="outRange")
        
        # get and convert range matrices for massaging
        eRInB, eRInT = self._shape_range_for_elementwise(self.inRange, rank=2)
        eROutB, eROutT= self._shape_range_for_elementwise(self.outRange, rank=3)
        
        #summary ops
        x_hist = tf.histogram_summary("x", fullIn)
        t_hist = tf.histogram_summary("t", fullOut_)
        #-----</Placeholder Construction>-----
        
        
        #-----<ANN Construction>-----
        with tf.name_scope('ANN_const') as scope:
            #two hidden layers
            
            w1 = self._weight_variable([inDims, hiddenLayerSize])
            b1 = self._bias_variable([hiddenLayerSize])
            
            w2 = self._weight_variable([hiddenLayerSize, hiddenLayerSize])
            b2 = self._bias_variable([hiddenLayerSize])
            
            w3 = self._weight_variable([hiddenLayerSize, outCount])
            b3 = self._bias_variable([outCount])
            
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
            lay1 = tf.tanh( (tf.matmul(netIn, w1)) + b1)
            #lay1 = tf.nn.bias_add(tf.matmul(netIn, w1), b1)
            lay2 = tf.tanh( (tf.matmul(lay1, w2)) + b2)
            #lay2 = tf.nn.bias_add(tf.matmul(lay1, w2), b2)
            netOut = tf.tanh( (tf.matmul(lay2, w3)) + b3)
            #netOut = tf.nn.bias_add(tf.matmul(lay2, w3), b3)
            
            #parse and shape netOut activations into coefficient arrays.
            mixRaw = tf.slice(netOut, begin=[0,0], size=[-1,g]) #.shape == [s,g]
            varRaw = tf.slice(netOut, begin=[0,g], size=[-1,outDims]) #.shape == [s,t]
            meanRaw = tf.reshape(tf.slice(netOut, begin=[0,g+outDims], size=[-1,-1]), shape=[-1,g,outDims]) #.shape == [s,g,t]
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
            mean = ((.5 + (meanRaw/2)) * (eROutT-eROutB)) + eROutB #this scales from relevant tanh output (-1,1)
            
        #summary ops
        mix_hist = tf.histogram_summary("Mixing Coefficients", mix)
        var_hist = tf.histogram_summary("Variances", var)
        mean_hist = tf.histogram_summary("Means", mean)
        #-----</Massage for Gaussian Mixture>-----
        
        #-----<Update Reference Dict>-----
        
        #FIXME this is a workaround for x retrievals that say, "fed and fetched, you suck"
        fetchX = ((.5 + (netIn/2)) * (eROutT-eROutB)) + eROutB  # expand from tanh
        mid_fetchT = (2*(fullOut_-eRInB)/(eRInT-eRInB)) - 1
        fetchT = ((.5 + (mid_fetchT/2)) * (eROutT-eROutB)) + eROutB  # expand from tanh
        
        #this section updates the instance dictionary that can be used from outside to run tensorflow output and to fill placeholders.
        rd = self.refDict
        rd['x']=fullIn
        rd['fetchX']=fetchX
        rd['t']=fullOut_
        rd['fetchT']=fetchT
        rd['u']=mean  # as in 'mew'
        rd['v']=var
        rd['m']=mix
        rd['inRange']=inRange
        rd['outRange']=outRange
        
        rd['mixRaw']=mixRaw
        rd['varRaw']=varRaw
        rd['meanRaw']=meanRaw
        
        rd['netIn']=netIn
        rd['lay1']=lay1
        rd['lay2']=lay2
        rd['netOut']=netOut
        
        rd['w1']=w1
        rd['w2']=w2
        rd['w3']=w3
        rd['b1']=b1
        rd['b2']=b2
        rd['b3']=b3
        rd['hiddenLayerSize']=tf.constant(hiddenLayerSize)
        #-----</Update Reference Dict>-----
        
        
    def _gmix_loss_nll(self):
        """Add the negative log likelihood loss tensor to the reference dictionary
            
            This defines the nll loss function for a gaussian mixture model.
            
            #for shape definitions, s is the sample size, g is the number of gaussian components, t is the number of target components)
            
           'm' is the mixing coefficent array of shape [s,g]
           'v' is the variance array of shape [s,t]
           'u' is the mean (mew) array of shape [s,g,t]
           't' is the target array of shape [s,t]
        """
        
        #collect tensors from reference dictionary
        try:
            rd = self.refDict
            m = rd['m']
            v = rd['v']
            u = rd['u']
            t = rd['t']
        except KeyError:
            raise KeyError('Forward model tensors are missing from the reference dictionary! Call .gmix_model before .gmix_nll')
            
            
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
            
        #-----<Update Reference Dict>-----
        self.refDict['loss_nll'] = nll
        #-----</Update Reference Dict>-----
        
    def _gmix_training_model(self, learningRate=None):
        """Adds general graph tensors to reference dictionary.
        
            - builds the mixture model that takes inputs and returns, means, variances, and mixture coefficients.
            - uses those values and builds the loss model.
            - builds a training step
            - initializes the session and variables
            - merges and preps summary writing
        """
        
        #-----<Argument Processing>-----
        inDims = self.inDims
        outDims = self.outDims
        if learningRate is None:
            learningRate = self.learningRate
        #-----</Argument Processing>-----
        
        with tf.name_scope('forward_model'):
            self._gmix_forward_model()
        
        with tf.name_scope('loss_model') as scope:
            self._gmix_loss_nll()
        
        with tf.name_scope('train_step') as scope:
            optimizer = tf.train.GradientDescentOptimizer(learningRate)
            gradients = optimizer.compute_gradients(self.refDict['loss_nll'])
            train_step = optimizer.minimize(self.refDict['loss_nll'])
        
        sess = tf.Session()
        
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter(SUMMARY_DIRECTORY, sess.graph_def)
        
        sess.run(tf.initialize_all_variables())
        
        #-----<Update Reference Dict>-----
        rd = self.refDict
        rd['sess']=sess
        rd['summaries']=merged
        rd['summaryWriter']=writer
        rd['train_step']=train_step
        rd['gradients']=gradients
        rd['train_step']=train_step
        #-----</Update Reference Dict>-----
    def _get_refDict(self):
        """Return the reference dictionary holding tensorflow tensors.
        """
        return self.refDict
    def _massage_training_arguments(self, iterations, testBatchSize, trainBatchSize):
        iterations = range(iterations)
        
        #verify batch sizes
        if self.x_test.shape[0] < testBatchSize:
            testBatchSize = self.x_test.shape[0]
        
        if self.x.shape[0] < trainBatchSize:
            trainBatchSize = self.x.shape[0]
        
        return iterations, testBatchSize, trainBatchSize
        
    def _tdb_nodes(self):
        """Add a list of tdb nodes for evaluation by tdb.debug
        """
        nodeList = []
        
        #----<Plots>----
        
        #mixing coefficients full update
        p_xm = tdb.plot_op(viz.mixing_coefficients, inputs=[
                                                self.graph.as_graph_element(self.refDict['fetchX']),
                                                self.graph.as_graph_element(self.refDict['m'])
                                            ])
        nodeList.append(p_xm)
        
        #variances full update
        p_xv = tdb.plot_op(viz.variances, inputs=[
                                                self.graph.as_graph_element(self.refDict['fetchX']),
                                                self.graph.as_graph_element(self.refDict['v'])
                                            ])
        nodeList.append(p_xv)
        
        #loss over iterations
        p_loss = tdb.plot_op(viz.watch_loss, inputs=[
                                                self.graph.as_graph_element(self.refDict['loss_nll'])
                                            ])
        nodeList.append(p_loss)
        
        #full 2d sampling of model.
        p_sample = tdb.plot_op(viz.sample, inputs=[
                                                self.graph.as_graph_element(self.refDict['fetchX']),
                                                self.graph.as_graph_element(self.refDict['m']),
                                                self.graph.as_graph_element(self.refDict['v']),
                                                self.graph.as_graph_element(self.refDict['u'])
                                            ])
        nodeList.append(p_sample)
        
        #sample means
        p_xu = tdb.plot_op(viz.means, inputs=[
                                                self.graph.as_graph_element(self.refDict['fetchX']),
                                                self.graph.as_graph_element(self.refDict['u'])
                                            ])
        nodeList.append(p_xu)
        
        #lay1
        p_lay1 = tdb.plot_op(viz.lay1_overX, inputs=[
                                                            self.graph.as_graph_element(self.refDict['fetchX']),
                                                            self.graph.as_graph_element(self.refDict['lay1'])
                                                        ])
        nodeList.append(p_lay1)
        
        #lay2
        p_lay2 = tdb.plot_op(viz.lay2_overX, inputs=[
                                                            self.graph.as_graph_element(self.refDict['fetchX']),
                                                            self.graph.as_graph_element(self.refDict['lay2'])
                                                        ])
        nodeList.append(p_lay2)
        
        #net out
        p_netout = tdb.plot_op(viz.netOut_overX, inputs=[
                                                                self.graph.as_graph_element(self.refDict['fetchX']),
                                                                self.graph.as_graph_element(self.refDict['netOut'])
                                                            ])
        nodeList.append(p_netout)
        
        #report on net calculations outside of tensorflow.
        p_ann = tdb.plot_op(viz.report_net, inputs=[
                                                                self.graph.as_graph_element(self.refDict['w1']),
                                                                self.graph.as_graph_element(self.refDict['b1']),
                                                                self.graph.as_graph_element(self.refDict['w2']),
                                                                self.graph.as_graph_element(self.refDict['b2']),
                                                                self.graph.as_graph_element(self.refDict['w3']),
                                                                self.graph.as_graph_element(self.refDict['b3']),
                                                                self.graph.as_graph_element(self.refDict['hiddenLayerSize'])
                                                            ])
        nodeList.append(p_ann)
        
        #training data
        p_training_data = tdb.plot_op(viz.training_data, inputs=[
                                                                self.graph.as_graph_element(self.refDict['fetchX']),
                                                                self.graph.as_graph_element(self.refDict['fetchT'])
                                                            ])
        nodeList.append(p_training_data)
        
        #----</Plots>----
        
        #----<Report ops>----
        
        
        #----</Report ops>----
        
        self.refDict['tdb_nodes'] = nodeList
    
    def train(self, iterations=1000, testBatchSize=500, trainBatchSize=1000, reportEvery=10):
        with self.graph.as_default():
            iterations, testBatchSize, trainBatchSize = self._massage_training_arguments(iterations, testBatchSize, trainBatchSize)
            
            #build feed dict, but leave x and t empty, as they will be updated in the training loop.
            feed_dict = {
                            self.refDict['x']:None,
                            self.refDict['t']:None,
                            self.refDict['inRange']:self.inRange,
                            self.refDict['outRange']:self.outRange
                        }
            
            result = None #final result for notebook testing.
            #----<Training Loop>----
            for i in iterations:
            
                if i % 10 == 0: #run reports every 10 iterations.
                    
                    #update feed_dict with test batch
                    feed_dict[self.refDict['x']], feed_dict[self.refDict['t']] = self._sample_batch(self.x_test, self.t_test, testBatchSize)
                    
                    evals = [
                                #self.refDict['summaries']#,
                                #self.refDict['loss_nll']
                                
                                #TEMP
                                self.refDict['w1'],
                                self.refDict['b1'],
                                self.refDict['w2'],
                                self.refDict['b2'],
                                self.refDict['w3'],
                                self.refDict['b3'],
                                self.refDict['lay1'],
                                self.refDict['lay2'],
                                self.refDict['netOut'],
                                self.refDict['netIn']
                            ]
                            
                    evals.extend(self.refDict['tdb_nodes']) #extend with the list of tensorflow debugger nodes
                    
                    status, result = tdb.debug(evals, feed_dict=feed_dict, session=self.refDict['sess'])
                    
                    #self.refDict['summaryWriter'].add_summary(result[0], i) #write to summary
                    #print("Loss at step %s: %s" % (i, result[1])) #print loss
                    #print '-------------------------------'
                    
                
                else:
                    #update feed_dict with training batch
                    feed_dict[self.refDict['x']], feed_dict[self.refDict['t']] = self._sample_batch(self.x, self.t, trainBatchSize)
                    self.refDict['sess'].run([
                                            self.refDict['train_step']
                                        ], feed_dict=feed_dict) #run train_step with training batch
            #----<Training Loop>----
            
            #TEMP for notebook testing.
            return dict( 
                        w1=result[0],
                        b1=result[1],
                        w2=result[2],
                        b2=result[3],
                        w3=result[4],
                        b3=result[5],
                        lay1=result[6],
                        lay2=result[7],
                        netOut=result[8],
                        netIn=result[9]
                        )
            
    def get_xmvu(self):
        with self.graph.as_default():
            feed_dict = {
                            self.refDict['x']:self.x_test,
                            self.refDict['t']:self.t_test,
                            self.refDict['inRange']:self.inRange,
                            self.refDict['outRange']:self.outRange
                        }
        
            #evaluate the trained model at m, v, and u with the test data.
            result = self.refDict['sess'].run([
                                            self.refDict['m'],
                                            self.refDict['v'],
                                            self.refDict['u']
                                        ],feed_dict=feed_dict)
            return result[0], result[1], result[2] #m, v, u
    def get_write_xmvu(self, path):
        m, v, u = self.get_xmvu()
        
        np.savez(path, x=self.t_test, m=m, v=v, u=u) #write this to an npz file for later sampling

#
#----------------------------------------------------------------<MAIN>----------------------------------------------------------------

"""
command line calling structure
:python thisfile.py <str:train.npz> <str:test.npz>
   where train.npz and test.npz should have two arrays, x and t.
   and x.shape == (s,inDims)
   and t.shape == (s,outDims)
train and test should obviously be different sets of data from the same source.

"""

XMVU_PATH = "/Users/azane/GitRepo/spider/data/xmvu.npz" #file to which the xmvu data, for sampling, is written after training.

if __name__ == "__main__":
    #TODO TODO split up this main section. make a function that can be called just with x and t array values
    #       cz the only reason to really use the command line is to read from a file...so that's really the only thing we need in the main section.
    
    #----<Read Data>----
    #t for test, s for training sample
    s_x, s_t = get_xt_from_npz(sys.argv[1])
    t_x, t_t = get_xt_from_npz(sys.argv[2])
    #----</Read Data>----
    
    gmm = GaussianMixtureModel(s_x, s_t, t_x, t_t)
    gmm.train()
    gmm.get_write_xmvu(XMVU_PATH)
    