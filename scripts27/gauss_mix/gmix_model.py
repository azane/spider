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
    
def tf_gaussian_likelihood(m, v, u, t):
    """Return the tf likelihood tensor
    
    This function builds a tf node that calculates the likelihood of t, given means and variance, as a function of the normal.
    This requires that m,v,u,t can all be broadcast together.
    """
    
    #TODO FIXME how does tf handle verification? like that mvut can be broadcast together!?
    
    with tf.name_scope('normal_likelihood') as scope:
        
        #prep terms of variance
        v_norm = 1/(v*tf.sqrt(2*np.pi))  # the normalizing term
        v_dem = 2*tf.square(v)  # the denominator in the exp
        
        #prep numerator term with corresponding sample target values and net proposed means
        tm_num = tf.square(t-u)
        
        #employ terms in pre-mixed likelihood function.
        premix = v_norm*(tf.exp(-(tm_num/v_dem)))
        
        #mix gaussian components
        #FIXME dkgni19dfohkjodnah9 didn't fix the flatlining problem...no difference in weight/bias gradient overload either.
        #dkgni19dfohkjodnah9 trying something. m error gradient is wrong, so use 1 component, and don't calculate it.
        likelihood = m*premix #disabled dkgni19dfohkjodnah9
        #likelihood = (m/m)*premix #dkgni19dfohkjodnah9
        
    return likelihood  # tf.shape(likelihood) == (s,g,t)

def tf_gmm_likelihood_k(m, v, u, t):
    """Return the probability that t came from g component
    
    This function builds a tf node that calculates the likelihood of t for individual components, over an entire mixture model.
    This divides the likelihood of each component by the sum of the others, thus normalizing the probability of each component.
    This requires that m,v,u,t can all be broadcast together.
    """
    
    likelihood = tf_gaussian_likelihood(m, v, u, t)
    
    #sum over the components for each sample, for each target
    denominator = tf.reduce_sum(likelihood, reduction_indices=1, keep_dims=True)  # tf.shape(denominator) = (s,1,t)
    
    #return the probability that t target came from g component
    return likelihood/denominator  # return.shape() == (s,g,t)

#----<Loss Gradients>-----
#   NOTE: v(standard deviation), u(means), and t(targets) must all be normalized to the same range before calling these functions.
#   FIXME verify that this is the case!!!
def tf_grad_mvu(m, v, u, t):
    """
    This is a helper function that takes the same mvut and calculates and returns all the gradients.
        - to prevent accidentally calling the grad functions wih mvut over differing ranges.
    This requires that m,v,u,t can all be broadcast together.
    """
    grad_m = tf_grad_m(m, v, u, t)
    grad_v = tf_grad_v(m, v, u, t)
    grad_u = tf_grad_u(m, v, u, t)
    
    return grad_m, grad_v, grad_u
    
def tf_grad_m(m, v, u, t):
    """Return the unaggregated (over samples) gradient of the nll with respect to ann outputs governing mixing coefficients.
    This requires that m,v,u,t can all be broadcast together.
    """
    
    #compute the gradients
    #   the 'posterior' subtracted from the 'prior' mixing coefficient.
    grads =  m - tf_gmm_likelihood_k(m, v, u, t)  # grads.shape() == (s,g,t)
    
    #aggregate over the targets.
    #   preserve s for visualizing over x
    grads = tf.reduce_sum(grads, reduction_indices=2)  # .shape() == (s,g)
    
    return grads  # .shape() == (s,g)
    
def tf_grad_v(m, v, u, t):
    """Return the unaggregated (over samples) gradient of the nll with respect to ann outputs governing variance.
    This requires that m,v,u,t can all be broadcast together.
    """
    
    #compute the gradients
    grads = -1*tf_gmm_likelihood_k(m, v, u, t)*((tf.pow((t-u), 2)/tf.pow(v, 3))-(1/v))
    
    #aggregate over the gaussian components, as variance is by t.
    #   preserve s for visualizing over x
    grads = tf.reduce_sum(grads, reduction_indices=1)  # .shape() == (s,t)
    
    return grads  # .shape() == (s,t)
    
def tf_grad_u(m, v, u, t):
    """Return the unaggregate (over samples) gradient of the nll with respect to ann outputs governing the means.
    This requires that m,v,u,t can all be broadcast together.
    """
    
    #compute the gradients
    grads = tf_gmm_likelihood_k(m, v, u, t)*((u-t)/tf.pow(v, 2))
    
    return grads  # shape == (s,g,t)

#----</Loss Gradients>-----

def to_tanh(vals, rBot, rTop):
    """Return vals normalized to tanh range.
    rBot and rTop must match the shape of vals coming in.
    """
    return ((2*(vals-rBot)/(rTop-rBot)) - 1)
    
    
def from_tanh(vals, rBot, rTop):
    """Return vals expanded from tanh range.
    """
    return ((.5 + (vals/2)) * (rTop-rBot)) + rBot
    
#-----</Helper Functions>-----

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
        
    def _expand_mvut(self, m, v, u, t):
        """Return the expanded m, v, u, and t for broadcasting together.
        """
        # m.shape == (s,g)
        # v.shape == (s,t)
        # u.shape == (s,g,t)
        # t.shape == (s,t)
        
        m = tf.expand_dims(m, 2)  # shape == (s,g,1)
        v = tf.expand_dims(v, 1)  # shape == (s,1,t)
        #u = u
        t = tf.expand_dims(t, 1)  # shape == (s,1,t)
        
        return m, v, u, t
    def _weight_variable(self, shape):
        """Return weight matrix
        This initializes weight matrices for the ANN in this model.
        """
        #initialize weights on a truncated normal distribution
        initial = tf.truncated_normal(shape, stddev=0.5)
        #tf.clip_by_value(initial, -5., 5.)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        """Return bias matrix
        This initializes bias matrices for the ANN in this model.
        """
        #initialize non-zero bias matrix
        initial = tf.constant(0.1, shape=shape)
        #tf.clip_by_value(initial, -1., 1.)
        return tf.Variable(initial)

    def _shape_range_for_elementwise(self, r, rank=1):
        """Return reshaped range matrices
        This shapes range matrices for easier calculations.
        """
    
        #r: the range matrix
        #    [ [rb, rt]
        #      [rb, rt]
        #      [rb, rt] ]
        r = tf.transpose(r)  # group rbot and rtop into rows
        #r:
        #    [ [rb, rb, rb]
        #      [rt, rt, rt] ]
        rb, rt = tf.split(0, 2, r)  # split range bottom and range top (rb, rt) into two tensors.
        #     rb = [rb, rb, rb]
        #     rt = [rt, rt, rt]
    
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
            
            ##TODO FIXME I'm going to need to calculate the weight and bias gradients...
            #               but that's all it seems to be getting wrong. the layer activations seem correct.
            #               from those, it should be easy to calculate stuff.
            
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
            #meanRaw = tf.reshape(tf.slice(netOut, begin=[0,g+outDims], size=[-1,g*outDims]), shape=[-1,g,outDims]) #.shape == [s,g,t]
        #-----</ANN Execution>-----
        
        
        #-----<Massage for Gaussian Mixture>-----
        #massaging functions, prep ANN outputs for use in gaussian mixture, see notes for explanation
        
        with tf.name_scope('massage_mix') as scope:
            #mixing coefficient
            #compute softmax so that the mixing coefficients sum to 1 across x.
            mix = tf.nn.softmax(mixRaw)
            
        with tf.name_scope('massage_var') as scope:
            
            #FIXME jhd838891hdjdlsl set var to 1 to see if it's causing the problem.
            #this doesn't make tf calculate a zero gradient? HOW!? is this being used elsewhere?
            #   apparently there's still a way to change var that matters?
            #varRaw = varRaw*0 #var should be 1
            
            #variances
            #varScale = tf.Variable(2.0) #TODO is this a good idea?
            var = tf.exp(tf.mul(varRaw, 1.0)) #keep variance positive, and relevant. this scales from tanh output (-1,1)
            
            
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
        
        #rd['mixRaw']=mixRaw
        #rd['varRaw']=varRaw
        #rd['meanRaw']=meanRaw
        
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
            
            #expand dims
            m, v, u, t = self._expand_mvut(m, v, u, t)
            likelihood = tf_gaussian_likelihood(m, v, u, t)
            
            #sum over the likilihood of each target and gaussian component, don't reduce over samples yet.
            #FIXME i know the gaussian components are supposed to be summed, but i'm not sure about the various targets?
            #FIXME      do i have to mix the targets like the gaussian components?
            tot_likelihood = tf.reduce_sum(likelihood, [1,2]) #sum over g and t dimensions.
            
            """38ndksl3ihk
            Trying something. remove the summation over the samples, and let this be aggregated over the weights in the end."""
            #take natural log of sum, then reduce over samples, then negate for the final negative log likelihood
            #nll = -tf.reduce_sum(tf.log(tot_likelihood)) #this reduces along the final dimension, so nll will be a scalar. #disabled: 38ndksl3ihk
            nll = -tf.log(tot_likelihood) #38ndksl3ihk
            
            #summary ops
            likelihood_hist = tf.histogram_summary("Post-mixed Likelihood", likelihood)
            tot_likelihood_hist = tf.scalar_summary("Average Likelihood Across Samples", tf.reduce_mean(tot_likelihood))
            #nll_hist = tf.scalar_summary("Negative Log Likelihood", nll) #disabled : 38ndksl3ihk
            nll_hist = tf.scalar_summary("Negative Log Likelihood", tf.reduce_sum(nll)) #38ndksl3ihk
            
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
        rd = self.refDict
        if learningRate is None:
            learningRate = self.learningRate
        #-----</Argument Processing>-----
        
        with tf.name_scope('forward_model'):
            self._gmix_forward_model()
        
        with tf.name_scope('loss_model') as scope:
            self._gmix_loss_nll()
        
        with tf.name_scope('calc_own_gradients') as scope:
            #calculate the gradients of the activations attached to mvu, but outside of tensorflow.
            self._calc_mvu_gradients()
            #calculate the weight and bias gradients given tf calculated activation errors
            self._calc_wb_gradients_from_tf_activations()
        
        with tf.name_scope('train_step') as scope:
            optimizer = tf.train.GradientDescentOptimizer(learningRate)
            #optimizer = tf.train.MomentumOptimizer(learningRate, 0.01)
            train_step = optimizer.minimize(rd['loss_nll'])
            #custom train step applies the calculated and aggregated gradients.
            #   to modify the construction of these gradients,
            #   simply modify self._calc_wb_gradients_from_tf_activations
            custom_train_step = [
                                    rd['b1'].assign_sub(rd['calc_agg_grad_b1']),
                                    rd['b2'].assign_sub(rd['calc_agg_grad_b2']),
                                    rd['b3'].assign_sub(rd['calc_agg_grad_b3']),
                                    
                                    rd['w1'].assign_sub(rd['calc_agg_grad_w1']),
                                    rd['w2'].assign_sub(rd['calc_agg_grad_w2']),
                                    rd['w3'].assign_sub(rd['calc_agg_grad_w3'])
                                ]
        
        sess = tf.Session()
        
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter(SUMMARY_DIRECTORY, sess.graph_def)
        
        sess.run(tf.initialize_all_variables())
        
        #----<Convert TF calculated Gradients>----
        #all variables
        #this returns a list of tuples
        gradients = optimizer.compute_gradients(self.refDict['loss_nll'])
        gradients = dict(gradients) #store list of tuples as dict
        inv_grad = dict((v,k) for k,v in gradients.iteritems()) #invert dict so tensor is the key
        
        #loss with respect to m,v,u
        #this returns a list of gradients.
        #this gives the summed gradients over x, holding the others equal.
        #   i.e. it's a scalar value that simply shows how much each one needs to be different than what it is.
        grad_mvu = tf.gradients(self.refDict['loss_nll'], [
                                                            self.refDict['m'],
                                                            self.refDict['v'],
                                                            self.refDict['u']
                                                        ])
                                                        
        
        #----</Convert TF calculated Gradients>----
        
        #-----<Update Reference Dict>-----
        rd['sess']=sess
        rd['summaries']=merged
        rd['summaryWriter']=writer
        rd['tf_train_step']=train_step
        rd['custom_train_step']=custom_train_step
        
        #use the network variables as keys for the gradients dictionary, store for later use.
        #TODO FIXME rename these to tf_grad_** for clarity.
        rd['grad_w1']=inv_grad[rd['w1']]
        rd['grad_b1']=inv_grad[rd['b1']]
        rd['grad_w2']=inv_grad[rd['w2']]
        rd['grad_b2']=inv_grad[rd['b2']]
        rd['grad_w3']=inv_grad[rd['w3']]
        rd['grad_b3']=inv_grad[rd['b3']]
        
        rd['tf_grad_m']=grad_mvu[0]
        rd['tf_grad_v']=grad_mvu[1]
        rd['tf_grad_u']=grad_mvu[2]
        #-----</Update Reference Dict>-----
        
    def _get_refDict(self):
        """Return the reference dictionary holding tensorflow tensors.
        """
        #TODO incorporate some method whereby arbitrarily selected elements of the reference dict are compute and returned as ndarrays.
        return self.refDict
    def _massage_training_arguments(self, iterations, testBatchSize, trainBatchSize):
        iterations = range(iterations)
        
        #verify batch sizes
        if self.x_test.shape[0] < testBatchSize:
            testBatchSize = self.x_test.shape[0]
        
        if self.x.shape[0] < trainBatchSize:
            trainBatchSize = self.x.shape[0]
        
        return iterations, testBatchSize, trainBatchSize
        
    def _calc_mvu_gradients(self):
        """Adds calculated (outside of tf) gradients of the output activations to the reference dict.
        These are not aggregated over the samples.
        """
        
        rd = self.refDict
        m = rd['m']
        v = rd['v']
        u = rd['u']
        t = rd['t']
        
        m, v, u, t = self._expand_mvut(m, v, u, t)
        
        grad_m, grad_v, grad_u = tf_grad_mvu(m, v, u, t)
        
        self.refDict['calc_grad_m_activations']=grad_m
        self.refDict['calc_grad_v_activations']=grad_v
        self.refDict['calc_grad_u_activations']=grad_u
        
    def _calc_wb_gradients_from_tf_activations(self):
        """Adds non tf calculated weight and bias gradients from the tf calculated activation gradients.
        Stores tf calculated layer gradients
        Stores self calculated bias gradients
        Stores self calculated weight gradients
        Stores aggregated versions of the weight and bias gradients
        """
        rd = self.refDict
        
        #calculate layer errors over s.
        grad_activations = tf.gradients(self.refDict['loss_nll'], [ 
                                                                    self.refDict['lay1'],
                                                                    self.refDict['lay2'],
                                                                    self.refDict['netOut']
                                                                ])
        #store in reference dict for more convenient naming.
        rd['grad_lay1']=grad_activations[0]  # .shape == (s,hSize)
        rd['grad_lay2']=grad_activations[1]
        rd['grad_netOut']=grad_activations[2]  # .shape == (s, netOutSize)
        
        
        #the bias gradients are the activation gradients. multiply by the learning rate first.
        rd['calc_grad_b1'] = rd['grad_lay1'] * self.learningRate
        rd['calc_grad_b2'] = rd['grad_lay2'] * self.learningRate
        rd['calc_grad_b3'] = rd['grad_netOut'] * self.learningRate
        
        #the weight gradients are the previous layer's activations multiplied by the error gradient in the output.
        #change so that ins.shape == (s,inSize,1) and outs.shape == (s,1,outSize)
        netIn_ = tf.expand_dims(rd['netIn'], 2)  # .shape == (s, inSize, 1)
        lay1_ = tf.expand_dims(rd['lay1'], 2)
        lay2_ = tf.expand_dims(rd['lay2'], 2)
        
        grad_lay1_ = tf.expand_dims(rd['grad_lay1'], 1)  # .shape == (s, 1, outSize)
        grad_lay2_ = tf.expand_dims(rd['grad_lay2'], 1)
        grad_netOut_ = tf.expand_dims(rd['grad_netOut'], 1)
        
        rd['calc_grad_w1'] = netIn_ * grad_lay1_ * self.learningRate
        rd['calc_grad_w2'] = lay1_ * grad_lay2_ * self.learningRate
        rd['calc_grad_w3'] = lay2_ * grad_netOut_ * self.learningRate
        
        #aggregate over samples with the average!
        agg_func = tf.reduce_mean
        rd['calc_agg_grad_b1'] = agg_func(rd['calc_grad_b1'], 0)
        rd['calc_agg_grad_b2'] = agg_func(rd['calc_grad_b2'], 0)
        rd['calc_agg_grad_b3'] = agg_func(rd['calc_grad_b3'], 0)
        
        rd['calc_agg_grad_w1'] = agg_func(rd['calc_grad_w1'], 0)
        rd['calc_agg_grad_w2'] = agg_func(rd['calc_grad_w2'], 0)
        rd['calc_agg_grad_w3'] = agg_func(rd['calc_grad_w3'], 0)
        
    def _tdb_nodes(self):
        """Add a list of tdb nodes for evaluation by tdb.debug
        """
        nodeList = []
        
        #----<Plots>----
        
        #weight error
        p_w1e = tdb.plot_op(viz.weights1, inputs=[
                                                self.graph.as_graph_element(self.refDict['fetchX']),
                                                self.graph.as_graph_element(self.refDict['calc_grad_w1'])
                                            ])
        nodeList.append(p_w1e)
        
        p_w2e = tdb.plot_op(viz.weights2, inputs=[
                                                self.graph.as_graph_element(self.refDict['fetchX']),
                                                self.graph.as_graph_element(self.refDict['calc_grad_w2'])
                                            ])
        nodeList.append(p_w2e)
        
        p_w3e = tdb.plot_op(viz.weights3, inputs=[
                                                self.graph.as_graph_element(self.refDict['fetchX']),
                                                self.graph.as_graph_element(self.refDict['calc_grad_w3'])
                                            ])
        nodeList.append(p_w3e)
        
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
        #nodeList.append(p_lay1)
        
        #lay2
        p_lay2 = tdb.plot_op(viz.lay2_overX, inputs=[
                                                            self.graph.as_graph_element(self.refDict['fetchX']),
                                                            self.graph.as_graph_element(self.refDict['lay2'])
                                                        ])
        #nodeList.append(p_lay2)
        
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
        #nodeList.append(p_ann)
        
        #training data
        p_training_data = tdb.plot_op(viz.training_data, inputs=[
                                                                self.graph.as_graph_element(self.refDict['fetchX']),
                                                                self.graph.as_graph_element(self.refDict['fetchT'])
                                                            ])
        nodeList.append(p_training_data)
        
        
        #----<Gradients Over X>-----
        #grad_m over x
        p_grad_m = tdb.plot_op(viz.calc_grad_m, inputs=[
                                                self.graph.as_graph_element(self.refDict['fetchX']),
                                                self.graph.as_graph_element(self.refDict['calc_grad_m_activations'])
                                            ])
        nodeList.append(p_grad_m)
        
        #grad_v over x
        p_grad_v = tdb.plot_op(viz.calc_grad_v, inputs=[
                                                self.graph.as_graph_element(self.refDict['fetchX']),
                                                self.graph.as_graph_element(self.refDict['calc_grad_v_activations'])
                                            ])
        nodeList.append(p_grad_v)
        
        #grad_u over x
        p_grad_u = tdb.plot_op(viz.calc_grad_u, inputs=[
                                                self.graph.as_graph_element(self.refDict['fetchX']),
                                                self.graph.as_graph_element(self.refDict['calc_grad_u_activations'])
                                            ])
        nodeList.append(p_grad_u)
        
        
        #grad_m over x
        p_tf_grad_m = tdb.plot_op(viz.tf_grad_m, inputs=[
                                                self.graph.as_graph_element(self.refDict['fetchX']),
                                                self.graph.as_graph_element(self.refDict['tf_grad_m'])
                                            ])
        nodeList.append(p_tf_grad_m)
        
        #grad_v over x
        p_tf_grad_v = tdb.plot_op(viz.tf_grad_v, inputs=[
                                                self.graph.as_graph_element(self.refDict['fetchX']),
                                                self.graph.as_graph_element(self.refDict['tf_grad_v'])
                                            ])
        nodeList.append(p_tf_grad_v)
        
        #grad_u over x
        p_tf_grad_u = tdb.plot_op(viz.tf_grad_u, inputs=[
                                                self.graph.as_graph_element(self.refDict['fetchX']),
                                                self.graph.as_graph_element(self.refDict['tf_grad_u'])
                                            ])
        nodeList.append(p_tf_grad_u)
        
        
        #grad_netOut over x
        p_tf_grad_netOut = tdb.plot_op(viz.tf_grad_netOut, inputs=[
                                                self.graph.as_graph_element(self.refDict['fetchX']),
                                                self.graph.as_graph_element(self.refDict['grad_netOut'])
                                            ])
        nodeList.append(p_tf_grad_netOut)
        #----<Gradients Over X>-----
        
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
            
                if i % reportEvery == 0: #run reports every 10 iterations.
                    
                    #update feed_dict with test batch
                    feed_dict[self.refDict['x']], feed_dict[self.refDict['t']] = self._sample_batch(self.x_test, self.t_test, testBatchSize)
                    
                    evals = [
                                #self.refDict['summaries']#,
                                #self.refDict['loss_nll']
                                
                                #network stuff
                                self.refDict['w1'], #0
                                self.refDict['b1'], #1
                                self.refDict['w2'], #2
                                self.refDict['b2'], #3
                                self.refDict['w3'], #4
                                self.refDict['b3'], #5
                                self.refDict['lay1'], #6
                                self.refDict['lay2'], #7
                                self.refDict['netOut'], #8
                                self.refDict['netIn'], #9
                                
                                #gradient stuff
                                self.refDict['grad_w1'], #10
                                self.refDict['grad_b1'], #11
                                self.refDict['grad_w2'], #12
                                self.refDict['grad_b2'], #13
                                self.refDict['grad_w3'], #14
                                self.refDict['grad_b3'], #15
                                
                                self.refDict['tf_grad_m'], #16
                                self.refDict['tf_grad_v'], #17
                                self.refDict['tf_grad_u'], #18
                                
                                self.refDict['calc_agg_grad_w1'], #19
                                self.refDict['calc_agg_grad_b1'], #20
                                self.refDict['calc_agg_grad_w2'], #21
                                self.refDict['calc_agg_grad_b2'], #22
                                self.refDict['calc_agg_grad_w3'], #23
                                self.refDict['calc_agg_grad_b3'] #24
                            ]
                            
                    evals.extend(self.refDict['tdb_nodes']) #extend with the list of tensorflow debugger nodes
                    
                    breakpoints = [
                                    self.refDict['loss_nll']
                                ]
                    
                    status, result = tdb.debug(evals, feed_dict=feed_dict, session=self.refDict['sess'])#, breakpoints=breakpoints)
                    
                    #self.refDict['summaryWriter'].add_summary(result[0], i) #write to summary
                    #print("Loss at step %s: %s" % (i, result[1])) #print loss
                    #print '-------------------------------'
                    
                
                else:
                    #update feed_dict with training batch
                    feed_dict[self.refDict['x']], feed_dict[self.refDict['t']] = self._sample_batch(self.x, self.t, trainBatchSize)
                    #custom train step is a list itself, the returned value is a list of variable values after the training.
                    self.refDict['sess'].run(self.refDict['custom_train_step'], feed_dict=feed_dict)
            #----<Training Loop>----
            
            #for notebook testing.
            return dict(
                        
                        #net stuff
                        w1=result[0],
                        b1=result[1],
                        w2=result[2],
                        b2=result[3],
                        w3=result[4],
                        b3=result[5],
                        lay1=result[6],
                        lay2=result[7],
                        netOut=result[8],
                        netIn=result[9],
                        
                        #gradient stuff
                        grad_w1=result[10],
                        grad_b1=result[11],
                        grad_w2=result[12],
                        grad_b2=result[13],
                        grad_w3=result[14],
                        grad_b3=result[15],
                        
                        tf_grad_m=result[16],
                        tf_grad_v=result[17],
                        tf_grad_u=result[18],
                        
                        calc_agg_grad_w1=result[19],
                        calc_agg_grad_b1=result[20],
                        calc_agg_grad_w2=result[21],
                        calc_agg_grad_b2=result[22],
                        calc_agg_grad_w3=result[23],
                        calc_agg_grad_b3=result[24]
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
    