import numpy as np
import tensorflow as tf

def softmax(x):
    # from https://gist.github.com/stober/1946926
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

class Explorer(object):
    def __init__(self, c_x):
        super(Explorer, self).__init__()
        
        #The explorers control feature x location, this excludes environmental features
        self._c_x = c_x
        
        self.value = 0
        
    def walk(self, stepVector):
        try:
            self._c_x += stepVector
        except TypeError:
            raise TypeError("Cannot add stepVector of type " + str(type(stepVector)) + " to explorer location np.ndarray.")
    
    def redrop(self, c_x):
        if c_x.shape != self._c_x.shape:
            raise ValueError("c_x must be the same shape as the existing position.")
        self._c_x = c_x
    
class ExplorerHQ(object):
    """The 'go between' class linking trained experiential models to the the spider's control features.
    
    certainty_func must be a function that takes calculated values from the reference dictionary of the forward graph,
        and calculates the difference between the two horizontal asymptotes of the integral of the squared probability distribution,
        for this x vector. From this, a form of standard deviation is calculated and divided by the sensor's range. This gives the
        percentage of the sensor range over which sensor values are likely to fall. This exactly reflects the certainty of any probability
        distribution in one value, such that it can be easily integrated into the point-value function.
        see https://www.desmos.com/calculator/zo2qkaol3j for more details on this.
    
    expectation_func must be a function that takes calculated values from the reference dictionary of the forward graph,
        and calculates the expected value of this distribution. This serves as the projected sensor value for this x vector.
    
    
    """
    
    #TODO if using a web server, maybe receive a pickled reference dict of the forward graph?
    def __init__(self, numExplorers, xRange, sRange, forwardRD, certainty_func, expectation_func, sensorGoal=None, modifiers=None):
        super(ExplorerHQ, self).__init__()
        
        
        #NOTE: self._xRange and self._sRange are not to be used as constants in tensorflow calculations.
        #       if this is done, then we cannot update the range values of the graph after init.
        
        #use xRange and yRange to define in/out dimensionality
        #[ [rb, rt]
        #  [rb, rt] ]
        assert xRange.shape[1] == 2
        self._xRange = xRange
        self._xDim = self._xRange.shape[0]
        
        #remember that 'sensors' are spit out as probability distributions.
        #    so it may be better to think of this as the 'sensor expected value' range.
        assert sRange.shape[1] == 2
        self._sRange = sRange
        self._sDim = self._sRange.shape[0]
        
        #sensorGoal is the goal vector for the sensors.
        self._sensorGoal = np.ones(self._sDim)
        if sensorGoal is not None:
            self.update_sensorGoal(sensorGoal)
        
        #modifiers is a dictionary of modifiers for the final output.
        #   update_modifiers softmaxes the values, relative to each other.
        self._modifiers = dict(C=None, T=None, S=None)
        if modifiers is None:
            self.update_modifiers(dict(
                C=np.random.random_sample()
                T=np.random.random_sample()
                S=np.random.random_sample()
            ))
        else:
            self.update_modifiers(modifiers)
            
        
        #the reference dictionary of tensors in the forward model.
        #   this should hold at least
        #   1. one placeholder tensor x, where x[:,-1] are the gratification term values
        #   2. all placeholder tensors representing the parameters of the distribution
        #       this will only be directly dealt with by functions passed to this class.
        #   3. and the graph object to which these tensors are attached.
        #   4. and the two placeholders sRange and xRange that define the in and out value ranges.
        self.forwardRD = forwardRD
        
        #this function takes tensors defining the hyperparameters of the output distribution
        # from forwardRD and returns a tensor computing the certainty of the distribution
        self._certainty_func = certainty_func
        
        #this function takes tensors defining the hyperparameters of the output distribution
        # from forwardRD and returns a tensor computing the expectation of the distribution
        self._expectation_func = expectation_func
        
        #temp? assume that the gratification term is the last input.
        self._gtermIndex = (self._xDim - 1)
        
        #the reference dictionary of point value elements
        self.pvRD = {}
        
        #create and drop explorers in the space
        self.explorers = [Explorer(np.zeros(self._xDim, dtype='float64') for i in range(numExplorers))]
        [self.drop_explorer(e) for e in self.explorers]
        
        self._bestExplorer = #TODO this will store the best sensor value as the explorers are stepped.
        
        #build point value calculation tensors
        #   fill pvRD
        self._build_solution_space()
    
    def update_sensorGoal(self, sensorGoal):
        assert sensorGoal.shape == self._sensorGoal.shape
        assert sensorGoal.dtype == sensorGoal.dtype
        self._sensorGoal = sensorGoal
    
    def update_modifiers(self, modifiers):
        """Computes the softmax of the passed modifiers.
        """
        try:
            arr = np.array((modifiers['C'], modifiers['T'], modifiers['S']), dtype=float)
        except KeyError:
            raise AttributeError("'modifiers' must have 'C', 'T', and 'S' defined.")
        except ValueError:
            raise AttributeError("'modifiers' must be convertable to floats.")
            
        mm = softmax(arr)
        
        self._modifiers['C'] = mm[0]
        self._modifiers['T'] = mm[1]
        self._modifiers['S'] = mm[2]
    
    def _certainty(self):
        """Returns value of the certainty given certainty strictness/leniency.
            This passes the forward reference dictionary of the experiential model to the
             certainty_func passed on initialization.
             certainty_func should always return the a tensor of the value of the indefinite integral
              of the squared probability distribution, as it approaches infinity.
             i.e. this is the difference between the two horizontal asymptotes of the integral(p(x)**2)
        """
        
        #---<Certainty>---
        #this section calculates the certainty, from 0-1, of predictions given the p(x)
        
        #the value of the integral of the squared p(x) as it approaches infinity.
        # bigI.shape should be (numExplorers, outDims)
        bigI = self._certainty_func(self.forwardRD)
        
        #andy's calculation of the 'std' given the bigI
        #NOTE: i used np functions here, because they are constants.
        std = 1/(2*np.sqrt(np.pi)*bigI)
        
        #use the sRange placeholder from the forward RD.
        absRange = tf.reduce_sum(self.forwardRD['sRange']*np.array([[-1,1]]), reduction_indices=1)
        
        #divide the std by the absRange, i.e. return the ratio of the range relevantly covered by the p(x)
        certainty = std/absRange
        #---</Certainty>---
        
        #---<Certainty Value>---
        #this section calculates the certainty value given the certainty
        
        #TODO this value should eventually be exposed to the spider, as a constant, so that it can determine how strict it wants to be with certainty.
        #       this modifies the steepness of the exp(). see https://www.desmos.com/calculator/etoiuwakda
        #       0>this<1. lower values are stricter, 'tighter' against the y axis.
        #   to be exposed, this will need to be added to self.pvRD
        certaintyLeniency = tf.constant(0.1)
        
        #NOTE: this value will always be between 0-1
        return tf.exp(-certainty/certaintyLeniency)
        
        #---</Certainty Value>---
        
        
    def _gratification_term(self):
        """Return the value of the gratification term.
            This allows the final solution value calculation to factor in the 'gratification term'
            i.e. the time until the solution value is realized.
        """
        #take size [-1, 1], i.e. all the rows, for that single column.
        gratificationTerm = tf.slice(self.forwardRD['x'], begin=[0,self._gtermIndex], size=[-1, 1])
        
        #TODO these values need to be exposed to the spider. see https://www.desmos.com/calculator/etoiuwakda for more info.
        #       gratificationTermRange is the abs value of the maximum gratification term likely to be seen.
        #        the values can go beyond that though, and not cause an issue.
        #       valueRangeEnd is the value of the gratification term at the end of gratificationTermRange.
        #        the size of this value determines just how little/much we care about gratification terms beyond
        #        the described maximum in gratificationRange. low means we don't care. high means we do.
        #        0.001 means we don't care. 0.1 means we care a lot. anything above that, and gratification range is probably too low.
        #       the purpose of this is so that we can eventually collect values on a distribution over time,
        #        so there's always a chance of higher values.
        #   to be exposed, these will need to be added to self.pvRD
        gratificationTermRange = tf.constant(25) #TODO infer this from the xRange placeholder in forwardRD?
        valueRangeEnd = tf.constant(0.001)
        
        #calculate the shaper based on the above values. 
        shaper = -tf.log(valueRangeEnd)/gratificationTermRange
        
        #NOTE: this value will always be between 0-1
        return tf.exp(-shaper*gratificationTerm)
        
        
    def _sensor(self):
        """Return value of the sensor.
            First get the squared error to the goal, then divide that by the squared range.
            Then, take this 0-1 value and calculate value for this error.
        """
        
        self.pvRD['sensorGoal'] = tf.placeholder(tf.float32, shape=[1, self._sDim])
        
        s = self._expectation_func(self.forwardRD)
        
        #the squared distance to the goal
        num = tf.square(self.pvRD['sensorGoal'] - s)
        #the difference between the top and the bottom, squared
        den = tf.square(tf.reduce_sum(self._sRange*np.array([[-1,1]]), reduction_indices=1))
        
        err = (num/den)
        
        #TODO this value should eventually be exposed to the spider, as a constant, so that it can determine how strict it wants to be with the sensor error.
        #       this modifies the steepness of the exp(). see https://www.desmos.com/calculator/etoiuwakda
        #       0>this<1. lower values are stricter, 'tighter' against the y axis.
        #   to be exposed, this will need to be added to self.pvRD
        errorLeniency = tf.constant(0.1)
        
        #NOTE this value will always be between 0-1
        return tf.exp(-err/errorLeniency)
        
        
    def _full_pointValue(self):
        """Return a tensor calculating the full point value.
            The derivative at explorer positions with respect to this value will determine their step direction.
        """
        
        #NOTE: remember, these modifiers are changed to sum to 1.
        #       when exp()s are multiplied by these modifiers,
        #       that sets the maximum value of the exp() to that modifier.
        #       thus, c+t+s can never exceed 1.
        self.pvRD['modifier_C'] = tf.placeholder(tf.float32, shape=[1,])
        self.pvRD['modifier_T'] = tf.placeholder(tf.float32, shape=[1,])
        self.pvRD['modifier_S'] = tf.placeholder(tf.float32, shape=[1,])
        
        c = self.pvRD['modifier_C']*self.pvRD['C']
        t = self.pvRD['modifier_T']*self.pvRD['T']
        s = self.pvRD['modifier_S']*self.pvRD['S']
        
        return c+t+s
        
        
    def _build_solution_space(self):
        with self.rd['graph'].as_default():
            self.pvRD['C'] = self._certainty()
            self.pvRD['T'] = self._gratification_term()
            self.pvRD['S'] = self._sensor()
            self.pvRD['V'] = self._full_pointValue()
        
    def drop_explorer(self, explorer):
        #TODO make more intelligent decisions about where to drop new explorers.
        drop = np.random.random_sample(size=self._xDim)
        #multiply the drop by the range, the difference between the range limits.
        drop *= np.sum((self._xRange*np.array([[-1,1]])), axis=1)
        #add the bottom of the range to the drop value
        drop += self._xRange[:,0]
        #redrop the explorer
        explorer.redrop(drop)
        
    def step_explorer(self):
        #TODO after the explorers take some steps, 
        pass
