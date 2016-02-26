import numpy as np
import tensorflow as tf

def softmax(x):
    # from https://gist.github.com/stober/1946926
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


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
    def __init__(self, numExplorers, xRange, sRange, forwardRD, certainty_func, expectation_func, sensorGoal=None, modifiers=None, forwardMapper=None):
        super(ExplorerHQ, self).__init__()
        
        #---<Forward Mapper>---
        #a dict used in referencing the forwardRD.
        # self.forwardRD[self.forwardMapper['xRange']] -> self.forwardRD['inRange']
        if forwardMapper is None:
            self.forwardMapper = dict(
                x='x',
                xRange='inRange',
                sRange='outRange',
                graph='graph'
            )
        else:
            try:
                forwardMapper['x']
                forwardMapper['xRange']
                forwardMapper['sRange']
                forwardMapper['graph']
            except KeyError:
                raise ValueError("The 'forwardMapper' dictionary must define at least 'x', 'xRange', 'sRange' and 'graph'")
                
        #the reference dictionary of tensors in the forward model.
        #   this should hold at least
        #   1. one placeholder tensor x, where x[:,-1] are the gratification term values
        #   2. all placeholder tensors representing the parameters of the distribution
        #       this will only be directly dealt with by functions passed to this class.
        #   3. and the graph object to which these tensors are attached.
        #   4. and the two placeholders sRange and xRange that define the in and out value ranges.
        self.forwardRD = forwardRD
        #---</Forward Mapper>---
        
        #---<Ranges>---
        #NOTE: self._xRange and self._sRange are not to be used as constants in tensorflow calculations.
        #       if this is done, then we cannot update the range values of the graph after init.
        #       instead, calculations referencing the ranges should reference xRange and sRange in the forwardRD
        
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
        #---</Ranges>---
        
        #---<Point Value Hyperparameters>---
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
        #---</Point Value Hyperparameters>---
        
        #---<Experiential Model Dependent Functions>---
        #this function takes tensors defining the hyperparameters of the output distribution
        # from forwardRD and returns a tensor computing the certainty of the distribution
        self._certainty_func = certainty_func
        
        #this function takes tensors defining the hyperparameters of the output distribution
        # from forwardRD and returns a tensor computing the expectation of the distribution
        self._expectation_func = expectation_func
        #---</Experiential Model Dependent Functions>---
        
        #---<Misc>---
        #temp? assume that the gratification term is the last input.
        self._gtermIndex = (self._xDim - 1)
        #---</Misc>---
        
        #---<Build TF Graph>---
        #the reference dictionary of point value elements
        self.pvRD = {}
        
        #create and drop explorers in the space
        with self.forwardRD[self.forwardMapper['graph']].as_default():
            self.explorers = tf.Variable(0., shape=[numExplorers, self._xDim], dtype=tf.float32)
            self.drop_explorer(range(numExplorers))
        
        #build point value calculation tensors
        #   fill pvRD
        self._build_solution_space()
        #---</Build TF Graph>---
    
    def update_sensorGoal(self, sensorGoal):
        """Updates sensorGoal attribute after verification
        """
        assert sensorGoal.shape == self._sensorGoal.shape
        assert sensorGoal.dtype == sensorGoal.dtype
        self._sensorGoal = sensorGoal
    
    def update_modifiers(self, modifiers):
        """Computes the softmax of the verified modifiers and updates attribute.
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
        """Returns tensor calculating the value of the certainty given certainty strictness/leniency.
        """
        
        #NOTE:
        #This passes the forward reference dictionary of the experiential model to the
        # certainty_func passed on initialization.
        # certainty_func should always return the a tensor of the value of the indefinite integral
        #  of the squared probability distribution, as it approaches infinity.
        # i.e. this is the difference between the two horizontal asymptotes of the integral(p(x)**2)
        
        #---<Certainty>---
        #this section calculates the certainty, from 0-1, of predictions given the p(x)
        
        #the value of the integral of the squared p(x) as it approaches infinity.
        # bigI.shape should be (numExplorers, outDims)
        bigI = self._certainty_func(self.forwardRD)
        
        #andy's calculation of the 'std' given the bigI
        #NOTE: i used np functions here, because the sqrt of pi is a constant.
        std = 1/(2*np.sqrt(np.pi)*bigI)
        
        #use the sRange placeholder from the forward RD.
        absRange = tf.reduce_sum(self.forwardRD[self.forwardMapper['sRange']]*np.array([[-1,1]]), reduction_indices=1)
        
        #divide the std by the absRange, i.e. return the ratio of the range relevantly covered by the p(x)
        certainty = std/absRange
        #---</Certainty>---
        
        #---<Certainty Value>---
        #this section calculates the certainty value given the certainty
        
        #TODO this value should eventually be exposed to the spider, as a constant, so that it can determine how strict it wants to be with certainty.
        #       this modifies the steepness of the exp(). see https://www.desmos.com/calculator/etoiuwakda
        #       0>this<1. lower values are stricter, 'tighter' against the y axis.
        #   to be exposed, this will need to be added to self.pvRD
        certaintyLeniency = tf.constant(0.1) #FIXME does this need to be a variable, placeholder?
        
        #NOTE: this value will always be between 0-1
        return tf.exp(-certainty/certaintyLeniency)
        
        #---</Certainty Value>---
        
        
    def _gratification_term(self):
        """Return valuation tensor of the gratification term.
            This allows the final solution value calculation to factor in the 'gratification term'
            i.e. the time until the solution value is realized.
        """
        #take size [-1, 1], i.e. all the rows, for that single column.
        gratificationTerm = tf.slice(self.forwardRD[self.forwardMapper['x']], begin=[0,self._gtermIndex], size=[-1, 1])
        
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
        valueRangeEnd = tf.constant(0.001) #FIXME does this need to be a variable, placeholder?
        
        #calculate the shaper based on the above values. 
        shaper = -tf.log(valueRangeEnd)/gratificationTermRange
        
        #NOTE: this value will always be between 0-1
        return tf.exp(-shaper*gratificationTerm)
        
        
    def _sensor(self):
        """Returns valuating tensor of the sensor estimation.
            First get the squared error to the goal, then divide that by the squared range.
            Then, take this 0-1 value and calculate value for this error.
        """
        
        self.pvRD['sensorGoal'] = tf.placeholder(tf.float32, shape=[self._sDim])
        #expand for broadcasting over explorers.
        tf.expand_dims(self.pvRD['sensorGoal'], 0)  # .shape == (1, self._sDim)
        
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
        errorLeniency = tf.constant(0.1) #FIXME does this need to be a variable, placeholder?
        
        #NOTE this value will always be between 0-1
        return tf.exp(-err/errorLeniency)
        
        
    def _explorer_isolation(self):
        #TODO create a mixture model of isotropically varying gaussians with means on the explorers.
        #       this will create an 'error' function that encourages explorers to travel away from each other.
        #       and will generally increase the uniqueness of solutions.
        
        #what if a grid approach is used, where explorers are made to stay in their grid?
        
        #what if an explorers 'grid' is simply a gaussian with mean of their initial drop?
        
        #also....i think we would only need one mixture function of positional gaussians.
        #   because, if each explorer sits perfectly atop their gaussian, the gradient with respect to their position will be 0.
        #   in other words, only other explorers can actually affect the gradient of this value, as only other explorer's gaussians
        #   will have means not centered on the explorer in question!
        
        #also, this isn't a probability distribution, so it doesn't have to sum to one. what if we square mixture?
        #   so that if two explorers pile up on each other, then that point is exponentially less favorable to other explorers.
        
        #what would we put as the variance? we'd use a diagonal, as we want the variance isotropic, i.e. centered in all directions.
        #   actually....maybe it shouldn't be centered? but "centered" if the ranges were normalized?
        #   yup. that's it.
        #   a mixture of gaussians, centered on each explorer, with their covariance such that the gaussian spreads over each x dimension at the same
        #    fraction of that x dimension's range.
        #   the magnitute of the covariance then, should be determined by the number of explorers and the dimensionality.
        #       so, 2 explorers on 1 dimension, would have a standard deviation of 1/4 of the range, so each covers half of the space?
        #       2 explorers in 2 dimensions, would have a standard deviation...
        
        #but how does this fit into the other point value elements? should the 
        pass
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
        """Calls all the graph building functions and stores in pvRD.
        """
        with self.forwardRD[self.forwardMapper['graph']].as_default():
            self.pvRD['C'] = self._certainty()
            self.pvRD['T'] = self._gratification_term()
            self.pvRD['S'] = self._sensor()
            self.pvRD['V'] = self._full_pointValue()
            
            self.pvRD['stepper'] = self._build_explorer_stepper()
            
            self.pvRD['sess'] = tf.Session()
            self.pvRD['sess'].run(tf.initialize_all_variables())
        
    def drop_explorer(self, explorerIndices):
        """Redrop explorers in the solution space
            explorerIndices are the indices of the rows for the explorers in self.explorers.
            self.explorers.assign (tf.Variable.assign) is used to assign a new array to the explorers.
        """
        
        #get current explorer positions
        npExplorers = self.explorers.eval(session=self.pvRD['sess'])
        
        #generate new drop points in range.
        drop = np.random.random_sample(size=(len(explorerIndices), self._xDim))
        drop *= np.sum((self._xRange*np.array([[[-1,1]]])), axis=2)
        drop += np.expand_dims(self._xRange[:,0],0)
        
        #update positions
        npExplorers[explorerIndices] = drop
        
        #reassign positions to tf variable.
        self.explorers.assign(npExplorers)
        
    def _build_explorer_stepper(self):
        """Returns a tensor that calculates the gradient of the point value and steps the explorers uphill.
            This also needs to integrate the explorers variable into the x placeholder process.
        """
        
        #FIXME 9dii10289hti ...i don't know how to to integrate the explorers...
        #       so i guess for now, we'll just calculate the gradient with respect to x.
        #       but i'm not sure you can calculate a gradient with respect to a placeholder?
        #calculate the gradient of the point values given explorer/x locations.
        explorerGrads = tf.gradient(self.pvRD['V'], self.forwardRD[self.forwardMapper['x']])
        
        #TODO expose this to the spider. make it a variable so it can change.
        rate = 1
        
        return self.explorers.assign_add(explorerGrads*rate)
    
    def step_explorer(self):
        """Evaluates the explorer stepper tensor to step explorers uphill.
        """
        feed_dict = {
                        self.forwardRD[self.forwardMapper['x']]:self.explorers, #FIXME i don't know if a tf.Variable can be fed into a placeholder? 9dii10289hti
                        self.forwardRD[self.forwardMapper['xRange']]:self._xRange,
                        self.forwardRD[self.forwardMapper['sRange']]:self._sRange,
                        
                        self.pvRD['sensorGoal']:self._sensorGoal,
                        self.pvRD['modifer_C']:self._modifiers['C'],
                        self.pvRD['modifer_T']:self._modifiers['T'],
                        self.pvRD['modifer_S']:self._modifiers['S']
                    }
        #FIXME 9dii10289hti . if not, then explorers can just be a numpy array...either way, it probably should be, i guess.
        
        #evaluate the stepper to step explorers uphill.
        self.pvRD['sess'].run([self.pvRD['stepper']], feed_dict=feed_dict)

def gmm_bigI(forwardRD):
    #NOTE: the sum rule takes the gaussian
def gmm_expectation(forwardRD):
    