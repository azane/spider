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
    """The 'go between' class linking trained experiential models to the the spider's control features."""
    
    #TODO if using a web server, maybe receive a pickled reference dict of the forward graph?
    def __init__(self, numExplorers, xRange, sRange, forwardRD, certainty_func, expectation_func, sensorGoal=None, modifiers=None):
        super(ExplorerHQ, self).__init__()
        
        
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
        """Returns the output values of the experiential model.
            This passes the reference dictionary of the experiential model to the
             certainty_func passed on initialization.
             certainty_func should always return the value of the indefinite integral
              of the squared probability distribution.
        """
        
        raw = self._certainty_func(self.forwardRD)
        
        
        
    def _gratification_term(self):
        """Return the time axis value
            This allows the final solution value calculation to factor in the 'gratification term'
            i.e. the time until the solution value is realized.
        """
        #take size [-1, 1], i.e. all the rows, for that single column.
        return tf.slice(self.forwardRD['x'], begin=[0,self._gtermIndex], size=[-1, 1])
        
    def _sensor(self):
        """Return the error between the sensor goal value and the distribution expectation.
            This is the squared error, divided by the sensor range.
            Thus, the output will always be between 0-1.
        """
        
        self.pvRD['sensorGoal'] = tf.placeholder(tf.float32, shape=[1, self._sDim])
        
        s = self._expectation_func(self.forwardRD)
        
        #the squared distance to the goal
        num = tf.square(self.pvRD['sensorGoal'] - s)
        #the difference between the top and the bottom
        den = tf.reduce_sum(self._yRange*np.array([[-1,1]]), reduction_indices=1)
        
        return (num/den)
        
    def _full_pointValue(self):
        """Return a tensor calculating the full point value.
            The derivative at explorer positions with respect to this value will determine their step direction.
        """
        
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
