import numpy as np
import tensorflow as tf

#FIXME what if a bimodal, uncertain prediction will yield better results in both modes than one unimodal element?

#FIXME Variable.assign* might need to be evaluated every time for it to take effect. in the explorer case,
#       we can't overwrite variable with the tensor returned by the assignment, otherwise the variableness is removed,
#       and nothing can be assigne.d

#FIXME instead of just adding the certainty:
#       a certaintly bad result is worse than an uncertain bad result. a certainly bad result is the worst thing!
#       a certainly good result is the best thing! an uncertain good result is better than a certain bad result.
#       implement this! so...certainty results in better points, the better the sensor value, and worse points the worse.

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
    def __init__(self, numExplorers, xRange, sRange, forwardRD, certainty_func, expectation_func, parameter_update_func,
                    sensorGoal=None, modifiers=None, forwardMapper=None, controlIndices=None):
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
        self._xRange = xRange.astype(np.float32, casting='same_kind')
        self._xDim = self._xRange.shape[0]
        
        #remember that 'sensors' are spit out as probability distributions.
        #    so it may be better to think of this as the 'sensor expected value' range.
        assert sRange.shape[1] == 2
        self._sRange = sRange.astype(np.float32, casting='same_kind')
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
                C=np.random.random_sample(),
                T=np.random.random_sample(),
                S=np.random.random_sample()
            ))
        else:
            self.update_modifiers(modifiers)
        #---</Point Value Hyperparameters>---
        
        #---<Experiential Model Dependent Functions>---
        #this function takes tensors defined the hyperparameters of the output distribution
        # from forwardRD and returns a tensor computing the certainty of the distribution
        self._certainty_func = certainty_func
        
        #this function takes tensors defined the hyperparameters of the output distribution
        # from forwardRD and returns a tensor computing the expectation of the distribution
        self._expectation_func = expectation_func
        
        #this function takes args and sets the trained parameters of the forwardRD
        #   it should return a list of variable assignment ops that are evaluated with the pv session
        #    upon their return to this the calling method.
        self._p_update_func = parameter_update_func
        #---</Experiential Model Dependent Functions>---
        
        #---<Misc>---
        #temp? assume that the gratification term is the last input.
        self._gtermIndex = (self._xDim - 1)
        
        self.update_controlIndices(controlIndices)
        #---</Misc>---
        
        #---<Build TF Graph>---
        #the reference dictionary of point value elements
        self.pvRD = {}
        
        #a reference dictionary for test values
        self.test = {}
        
        #create and drop explorers in the space
        with self.forwardRD[self.forwardMapper['graph']].as_default():
            #self.pvRD['sess'] = tf.Session()
            #self.explorers = tf.Variable(tf.zeros([numExplorers, self._xDim]))
            self.explorers = np.zeros([numExplorers, self._xDim])
            #self.pvRD['sess'].run(tf.initialize_all_variables())
            self.drop_explorer(range(numExplorers))
        
        #build point value calculation tensors
        #   fill pvRD
        #FIXME 89991jdkdlsnhj1h1 spider brain needs to initalize this class after it's figured out its nodes.
        #       for now, worlds is doing it, but that means this has to be called after spider has initialized and this class has.
        #self._build_solution_space()
        #---</Build TF Graph>---
    
    def update_controlIndices(self, controlIndices):
        """Updates the controlIndice array.
        """
        if controlIndices is not None:
            assert controlIndices.ndim == 1
            assert controlIndices.dtype == int
        self._controlIndices = controlIndices
        
    def update_sensorGoal(self, sensorGoal):
        """Updates sensorGoal attribute after verification
        """
        assert sensorGoal.shape == self._sensorGoal.shape
        assert sensorGoal.dtype == self._sensorGoal.dtype
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
        absRange = tf.reduce_sum(self.forwardRD[self.forwardMapper['sRange']]*np.array([[-1,1]]), reduction_indices=1)  # .shape == (sDim)
        absRange = tf.expand_dims(absRange, 0)  # .shape == (1,sDim) # for broadcasting over e.
        
        #divide the std by the absRange, i.e. return the ratio of the range relevantly covered by the p(x)
        certainty = std/absRange
        #---</Certainty>---
        
        #---<Certainty Value>---
        #this section calculates the certainty value given the certainty
        
        #TODO this value should eventually be exposed to the spider, as a constant, so that it can determine how strict it wants to be with certainty.
        #       this modifies the steepness of the exp(). see https://www.desmos.com/calculator/etoiuwakda
        #       0>this<1. lower values are stricter, 'tighter' against the y axis.
        #   to be exposed, this will need to be added to self.pvRD
        certaintyLeniency = tf.constant(0.25) #FIXME does this need to be a variable, placeholder?
        
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
        
        #absolute the grat term.
        gratificationTerm = tf.abs(gratificationTerm)
        
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
        gratificationTermRange = tf.constant(25.) #TODO infer this from the xRange placeholder in forwardRD?
        valueRangeEnd = tf.constant(0.1) #FIXME does this need to be a variable, placeholder?
        
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
        num = tf.square(self.pvRD['sensorGoal'] - s)  # .shape == (e, sDim)
        #the difference between the top and the bottom, squared
        den = tf.square(tf.reduce_sum(self.forwardRD[self.forwardMapper['sRange']]*np.array([[-1.,1.]], dtype=np.float32), reduction_indices=1))  # .shape == (sDim)
        den = tf.expand_dims(den, 0)  # .shape == (1,sDim) # for broadcasting over e.
        
        self.test['errDen'] = den
        self.test['errNum'] = num
        self.test['sensorVal'] = s
        
        
        err = (num/den)
        
        #TODO this value should eventually be exposed to the spider, as a constant, so that it can determine how strict it wants to be with the sensor error.
        #       this modifies the steepness of the exp(). see https://www.desmos.com/calculator/etoiuwakda
        #       0>this<1. lower values are stricter, 'tighter' against the y axis.
        #   to be exposed, this will need to be added to self.pvRD
        errorLeniency = tf.constant(0.1) #FIXME does this need to be a variable, placeholder?
        
        #NOTE this value will always be between 0-1
        #FIXME return tf.exp(-err/errorLeniency)  # disabled for testing 1820gndksli
        #return s #FIXME enabled for testing 1820gndksli. this is the sensor expectation.
        
        return tf.exp(-err/errorLeniency) # temp reenabled 1820gndksli
        
        
    def _explorer_isolation(self):
        """Build a tensor that is exponentially steep where explorers are concentrated.
            This should never be evaluated with lots of explorers.
        """
        #NOTE: tensorflow indexing SUCKS. such a pain. AADSFHEOIPHEFOPUHAEFJBADSH #disabled 1928tskdhslj
        #print "fullControlIndices tensor shape: " + str([None, self._controlIndices.shape[0]])
        #self.pvRD['fullControlIndices'] = tf.placeholder(tf.int32, shape=[None, self._controlIndices.shape[0]])  # .shape == (e,numCF)
        ##gather using a non-broadcasting index. i couldn't just transpose and gather for some reason, it messed up the None shape in the placeholder?
        #conIns = tf.gather(self.forwardRD[self.forwardMapper['x']], self.pvRD['fullControlIndices'])
        #self.pvRD['conInputs'] = conIns  # store for later gradient testing.
        
        
        #TODO i think this will need to be a covariance matrix, taking values only on the diagonal,
        #   but that scales with the range of the control features.
        isolationWidth = tf.constant(30., dtype=tf.float32)
        
        #make a unique placeholder.
        assert self.explorers.shape[0] < 500, "Don't run the isolation gradients with over 500 explorers. You can change this limit if you want."
        self.pvRD['explorers_for_isolation'] = tf.placeholder(tf.float32, shape=[self.explorers.shape[0], self._xDim])  # .shape == (e,xDim)
        X_r = tf.expand_dims(self.pvRD['explorers_for_isolation'], 0)  # .shape == (1,e,xDim)
        X_c = tf.expand_dims(self.pvRD['explorers_for_isolation'], 1)  # .shape == (e,1,xDim)
        
        #calculate the euclidian distance between 'centers'
        #FIXME start out with abs, as i think the matrix square doesn't guarantee all positive?
        diffs = tf.abs(X_r - X_c)  # .shape == (e,e,xDim)
        
        self.test['diffs'] = diffs
        
        #FIXME is it supposed to be (e,e,xDim) times its transpose, or (e,xDim,e) times its transpose?
        diffs_T = tf.transpose(diffs, perm=[0,2,1])  # .shape == (e,xDim,e)
        unSummed = tf.batch_matmul(diffs, diffs_T)  # .shape == (e,e,e)
        
        self.test['unSummed'] = unSummed
        
        #don't sqrt this, as we'd square it in the eCurve anyway.
        #FIXME 18250dksh not the best method i think...but add 1e-10 to keep it from being 0, cz e^-0 is inf.
        distSqrd = tf.reduce_sum(unSummed, reduction_indices=2)+1e-5  # .shape == (e,e)
        
        self.test['distSqrd'] = distSqrd
        
        eCurve_unProd = 1. + tf.exp(-(distSqrd)/(2.*tf.square(isolationWidth)))  # .shape == (e,e)
        #eCurve_unSum = tf.exp(-(distSqrd)/(2.*tf.square(isolationWidth)))  # .shape == (e,e)
        
        self.test['eCurve_unProd'] = eCurve_unProd
        
        eCurve = tf.reduce_prod(eCurve_unProd, reduction_indices=1) - 1.  # .shape == (e,)
        #eCurve = tf.reduce_sum(eCurve_unSum, reduction_indices=1)  # .shape == (e,)
        
        return eCurve
        
    def _full_pointValue(self):
        """Return a tensor calculating the full point value.
            The derivative at explorer positions with respect to this value will determine their step direction.
        """
        
        #NOTE: remember, these modifiers are changed to sum to 1.
        #       when exp()s are multiplied by these modifiers,
        #       that sets the maximum value of the exp() to that modifier.
        #       thus, c+t+s can never exceed 1.
        self.pvRD['modifier_C'] = tf.placeholder(tf.float32, shape=())
        self.pvRD['modifier_T'] = tf.placeholder(tf.float32, shape=())
        self.pvRD['modifier_S'] = tf.placeholder(tf.float32, shape=())
        
        c = self.pvRD['modifier_C']*self.pvRD['C']  # shape == (e,s)
        #c = (1 - self.pvRD['modifier_C'])*self.pvRD['C']  # shape == (e,s)
        t = self.pvRD['modifier_T']*self.pvRD['T']  # shape == (e,1)
        s = self.pvRD['modifier_S']*self.pvRD['S']  # shape == (e,s)
        #s = (1 - self.pvRD['modifier_S'])*self.pvRD['S']  # shape == (e,s)
        
        self.test['modC'] = c
        self.test['modT'] = t
        self.test['modS'] = s
        
        #TEMP we'll want a more sophisticated, modifiable way for weighting the value of different sensors.
        c = tf.reduce_mean(c, 1)
        t = tf.squeeze(t)
        s = tf.reduce_mean(s, 1)
        
        return c+t+s  # .shape == (e,)
        #return c*s+t
        
        
    def _build_solution_space(self):
        """Calls all the graph building functions and stores in pvRD.
        """
        with self.forwardRD[self.forwardMapper['graph']].as_default():
            self.pvRD['sess'] = tf.Session()
            
            self.pvRD['C'] = self._certainty()
            self.pvRD['T'] = self._gratification_term()
            self.pvRD['S'] = self._sensor()
            self.pvRD['V'] = self._full_pointValue()
            self.pvRD['I'] = self._explorer_isolation()
            
            self.pvRD['reportGrad'], self.pvRD['crawlGrad'], self.pvRD['isolationGrad'] = self._build_explorer_stepper()
            
            self.pvRD['sess'].run(tf.initialize_all_variables())
        
    def drop_explorer(self, explorerIndices):
        """Redrop explorers in the solution space
            explorerIndices are the indices of the rows for the explorers in self.explorers.
            self.explorers.assign (tf.Variable.assign) is used to assign a new array to the explorers.
        """
        
        #get current explorer positions
        #npExplorers = self.explorers.eval(session=self.pvRD['sess'])
        npExplorers = self.explorers
        
        #generate new drop points in range.
        drop = np.random.random_sample(size=(len(explorerIndices), self._xDim))
        drop *= np.sum((self._xRange*np.array([[[-1,1]]])), axis=2)
        drop += np.expand_dims(self._xRange[:,0],0)
        
        #update positions
        npExplorers[explorerIndices] = drop
        
        #reassign positions to tf variable.
        #self.explorers.assign(npExplorers)
        self.explorers = npExplorers
        
    def _build_explorer_stepper(self):
        """Returns a tensor that calculates the gradient of the point value and steps the explorers uphill.
            This also needs to integrate the explorers variable into the x placeholder process.
        """
        
        #FIXME kdngio129fgs need to find a way to only calculate the gradients of the control features. use tf.dynamic_partition?
        
        firstGrad = tf.gradients(self.pvRD['V'], self.forwardRD[self.forwardMapper['x']])
        secondGrad = tf.gradients(firstGrad[0], self.forwardRD[self.forwardMapper['x']])
        
        #we can't do this method on the above? because the gradients depend on the x values of the environment features
        isolationGrad = tf.gradients(self.pvRD['I'], self.pvRD['explorers_for_isolation'])
        #isolationGrad = self.pvRD['I']
        
        #TODO expose this to the spider. make it a variable so it can change.
        rate = 3.
        
        #return the rated gradients for addition to explorers
        #   the explorers should crawl along the second, and report their vectors along the first.
        #   TEMP? trying out two kinds of explorers, so *rate both.
        return (firstGrad[0]*rate), (secondGrad[0]*rate), (isolationGrad[0]*rate)
    
    def step_explorer(self):
        """Evaluates the explorer stepper tensor to step explorers uphill.
        """
        
        #FIXME 39ndkslwo make conIndices a class attribute
        
        #FIXME just use attribute in this method.
        conIndices = self._controlIndices
        
        #get full controlIndex, non broadcasting.  #FIXME disabled 1928tskdhslj
        #   make empty array
        #fullControlIndices = np.zeros((self.explorers.shape[0], self._controlIndices.shape[0]), dtype=int)
        #   fill all rows with control indices
        #fullControlIndices[:] = self._controlIndices
        
        #print "fullControlIndices.shape " + str(fullControlIndices.shape)
        #print "self._controlIndices.shape " + str(self._controlIndices.shape)
        #print "self.explorers.shape " + str(self.explorers.shape)
        #print
        #end disabled 1928tskdhslj
        
        feed_dict = {
                        
                        self.forwardRD[self.forwardMapper['x']]:self.explorers,
                        self.pvRD['explorers_for_isolation']:self.explorers,
                        self.forwardRD[self.forwardMapper['xRange']]:self._xRange,
                        self.forwardRD[self.forwardMapper['sRange']]:self._sRange,
                        
                        #self.pvRD['fullControlIndices']:fullControlIndices,  # FIXME disabled 1928tskdhslj
                        self.pvRD['sensorGoal']:self._sensorGoal,
                        self.pvRD['modifier_C']:self._modifiers['C'],
                        self.pvRD['modifier_T']:self._modifiers['T'],
                        self.pvRD['modifier_S']:self._modifiers['S']
                    }
        #evaluate the stepper to step explorers uphill.
        
        #results = self.pvRD['sess'].run([self.pvRD['stepper']], feed_dict=feed_dict)[0]
        
        for i in range(1):
            results = self.pvRD['sess'].run([self.pvRD['reportGrad'], self.pvRD['crawlGrad'], self.pvRD['isolationGrad'], self.pvRD['V']], feed_dict=feed_dict)
            
            #the first gradient, for reporting its gradient vector
            reportGrad = results[0]
            #the second gradient, for crawling
            crawlGrad = results[1]
            #the gradient that should be subtracted from the movement to encourage isolation.
            isolationGrad = results[2]
            #the value of the original function
            explorerValue = results[3]
            
            #TEMP have half the explorers ascend 1st and the other 2nd gradients. dkwig90101kdnfko
            #t_numE = self.explorers.shape[0]
            #t_half = int(t_numE/2)
            t_half = 0
            
            #only step along control indices #FIXME kdngio129fgs won't have to index crawlGrad, as it should only contain control features
            #TEMP slice half dkwig90101kdnfko
            self.explorers[:t_half,conIndices] += crawlGrad[:t_half,conIndices] - isolationGrad[:t_half,conIndices]
            self.explorers[t_half:,conIndices] += reportGrad[t_half:,conIndices]*50 - isolationGrad[t_half:,conIndices]
            
            if not hasattr(self, '_explorerSeries'):
                self._explorerSeries = []
            if not hasattr(self, '_explorerGrads'):
                self._explorerGrads = []
            if not hasattr(self, '_explorerVals'):
                self._explorerVals = []
            if not hasattr(self, '_explorerBest'):
                self._explorerBest = []
            self._explorerSeries.append(np.copy(self.explorers))
            self._explorerGrads.append(reportGrad)
            self._explorerVals.append(explorerValue)
            self._explorerBest.append(np.copy(self.explorers[np.argmax(self._explorerVals[-1])]))
            
            #TEMPS
            if not hasattr(self, 'test_actuals'):
                self.test_actuals = {}
            #
            testers = self.pvRD['sess'].run([self.test['diffs'], self.test['unSummed'], self.test['eCurve_unProd'],
                                                self.test['distSqrd']], feed_dict=feed_dict)
            self.test_actuals['diffs'] = testers[0]
            self.test_actuals['unSummed'] = testers[1]
            self.test_actuals['eCurve_unProd'] = testers[2]
            self.test_actuals['distSqrd'] = testers[3]
            self.test_actuals['isolationGrad'] = isolationGrad
            #TEMPS
            
            #TODO now that we are using the second derivative to crawl, the explorer isolation can be more effectively used, and won't push
            #       keep explorers on the hillsides...wait. what if we do that, keep explorers on the hillsides.....but not everything is a circle...
            #       i dunno.
            
            #redrop relatively poorest performing explorer. (or many of them? maybe all below the mean?)
            #   not the slowest crawler, but the one on the not stepest hill
            #normalize to 1, by each other.
            cg_tot = np.sum(np.copy(reportGrad[:t_half,conIndices]), axis=0)  # .shape == (1,numCF)
            cg_contribution = reportGrad[:t_half,conIndices]/cg_tot  # (e,numCF)/(1,numCF) == (e,numCF)
            #get means of normalized explorers' gradients
            cg_means = np.mean(cg_contribution, axis=1)  # .shape == (e,)
            if cg_means.size > 0:
                cg_worstIndex = np.argmin(cg_means)
                self.drop_explorer([cg_worstIndex])
            #FIXME reading the previous gradient will be inaccurate for just dropped explorers.
            
            if explorerValue[t_half:].size > 0:
                #drop three lowest.
                rg_worstIndex = explorerValue[t_half:].argsort()[:3]#np.argmin(explorerValue[t_half:])
                self.drop_explorer(rg_worstIndex)
        
    def update_params(self, *args, **kwargs):
        """Build parameter updating ops for forward model.
        """
        assigners = self._p_update_func(self.forwardRD, *args, **kwargs)
        
        #evaluate the assigners so the vars get updated
        self.pvRD['sess'].run(assigners)
        
    def update_environs_only(self, x):
        """Update only environmental values in explorer vectors.
            This method is used when previously dictated control features may not be fully set in the spider.
                This occurs in direct control features if those are also controlled by a human.
                This occurs in abstracted 'control features' (sensors) when setting a control feature does not necessarily mean
                    that that control feature will respond immediately.
                
            'envIndices' (environmental features indices) or 'conIndices' (control feature indices) must be defined
                they should be array like and hold the indices of the respective feature types
        """
        
        #FIXME 39ndkslwo make conIndices a class attribute
        
        #NOTE: remember that explorers only move a little bit each time, but they chase maximums that move slowly as well.
        #       if their position is overwritten, then they may lose progress on the maximum they are chasing.
        #        also, if they are overwritten with current control features, then they will all be forced to one position.
        #       remember that the actual control feature settings are not necessarily considered by explorers.
        #       their position is considered. if the actual control feature setting is deemed helpful, then it should
        #       be recorded as a node as well.
        #       FIXME 18691019djdks i think this means that the nodes should not bother returning the values of their control features.
        
        #FIXME verify that x is the right shape for explorers
        assert x.shape == self.explorers.shape[1:]
        
        #FIXME just use attribute in this method.
        conIndices = self._controlIndices
        envIndices = None
        
        #create base partition array
        ePartition = np.zeros(self._xDim)
        
        #update with indices.
        if (envIndices is not None) and (conIndices is not None):
            raise ValueError("Only one of envIndices and conIndices should be defined.")
        elif conIndices is not None:
            ePartition[conIndices] = 1  # set to control partition
            ePartition = np.logical_not(ePartition)  # negate control partition
        elif envIndices is not None:
            ePartition[envIndices] = 1
        else:
            raise ValueError("Either envIndices or conIndices must be defined.")
        
        #maskify
        ePartition = ePartition.astype(bool)
        
        #get environs from x
        environs = x[ePartition]
        
        self.explorers[:,ePartition] = environs
        
    def graph_space(self, x):
        """Takes inputs, and generates the point values for those inputs. Also evaluates ops in self.test
        """
        feed_dict = {
                        #FIXME can't feed variable to placeholder, just make explorers a numpy array. 9dii10289hti
                        self.forwardRD[self.forwardMapper['x']]:x,
                        self.forwardRD[self.forwardMapper['xRange']]:self._xRange,
                        self.forwardRD[self.forwardMapper['sRange']]:self._sRange,
                        
                        self.pvRD['sensorGoal']:self._sensorGoal,
                        self.pvRD['modifier_C']:self._modifiers['C'],
                        self.pvRD['modifier_T']:self._modifiers['T'],
                        self.pvRD['modifier_S']:self._modifiers['S']
                    }
        #FIXME 9dii10289hti . if not, then explorers can just be a numpy array...either way, it probably should be, i guess.
        #                        i mean, if we have to say variable.eval() then we should just store it as a numpy array.
        
        #evals
        evals = [self.pvRD['V'], self.pvRD['C'], self.pvRD['T'], self.pvRD['S']]  # list pv elements first.
        # test evals.
        evals.extend([
                        self.test['errDen'],
                        self.test['errNum'],
                        self.test['sensorVal'],
                        
                        self.forwardRD['w1'],
                        
                        self.forwardRD['m'],
                        self.forwardRD['v'],
                        self.forwardRD['u'],
                        
                        self.test['modC'],
                        self.test['modT'],
                        self.test['modS']
                    ])
        
        #evaluate
        result = self.pvRD['sess'].run(evals, feed_dict=feed_dict)
        
        return x, result[0], result[1], result[2], result[3], result[4:]

#---<GMM helpers>---
def _fix_mvu(m, v, u):
    """Shape mvu coming out of forwardRD so it makes more sense.
    """
    
    # m.shape == (e,g)
    # v.shape == (e,t)
    # u.shape == (e,g,t)
    
    m = tf.expand_dims(m, 2)  # shape == (e,g,1)
    v = tf.expand_dims(v, 1)  # shape == (e,1,t)
    #u = u  # shape == (e,g,t)
    
    return m, v, u
    
def gmm_bigI(forwardRD):
    """Compute the bigI of a gaussian mixture model, given mvu.
        See wiki page at https://github.com/azane/spider/wiki/Certainty-of-a-Gaussian-Mixture-Model for details.
    """
    
    m, v, u = _fix_mvu(forwardRD['m'], forwardRD['v'], forwardRD['u'])
    
    #e-number of explorers or samples
    #g-number of gaussian components
    #t-number of output/target dimensions
    
    #get mvu and add dim for .shape == (e, g, t, 1)
    m1 = tf.expand_dims(m, 3)  # shape == (e, g, 1, 1)
    v1 = tf.expand_dims(v, 3)  # shape == (e, 1, t, 1)
    u1 = tf.expand_dims(u, 3)  # shape == (e, g, t, 1)
    
    #get transpose of the components for .shape == (e, 1, t, g)
    m2 = tf.transpose(m1, perm=[0,3,2,1])
    v2 = tf.transpose(v1, perm=[0,3,2,1])
    u2 = tf.transpose(u1, perm=[0,3,2,1])
    
    #compute the bigI of each pair
    ssv = tf.square(v1) + tf.square(v2)  # sum of squared variance
    den = 1/(np.sqrt(2*np.pi)*tf.sqrt(ssv))
    num = tf.exp(-tf.square(u1-u2)/(2*ssv))
    unSummed = (m1*m2)*(num/den)
    
    #sum over these elements over the components for the total bigI
    return tf.reduce_sum(unSummed, [1,3])  # from .shape == (e, g, t, g) to .shape == (e,t)
    
def gmm_expectation(forwardRD):
    """
        The sum of m*u
    """
    m, v, u = _fix_mvu(forwardRD['m'], forwardRD['v'], forwardRD['u'])
    
    #sum over the components.
    return tf.reduce_sum((m*u), reduction_indices=[1])  # from shape == (e,g,t) to shape == (e,t)
    
def gmm_p_updater(forwardRD, w1, b1, w2, b2, w3, b3):
    """Assigns trained parameters to the gmm forward model.
    """
    #remember, these are ops, not actual assignments.
    #   so now, when these are evaluated, it will perform the assignment.
    assigners = [
        forwardRD['w1'].assign(w1),
        forwardRD['b1'].assign(b1),
        forwardRD['w2'].assign(w2),
        forwardRD['b2'].assign(b2),
        forwardRD['w3'].assign(w3),
        forwardRD['b3'].assign(b3)
    ]
    return assigners
    
#---<GMM helpers>---