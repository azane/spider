import numpy as np

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
    """The base 'go between' class linking trained experiential models to the the spider's control features."""
    
    #TODO if using a web server, maybe receive a pickled reference dict of the forward graph?
    def __init__(self, numExplorers, expRange):
        super(ExperientialModel, self).__init__()
        
        #[ [rb, rt]
        #  [rb, rt] ]
        assert expRange.shape[1] == 2
        self._expRange = expRange
        self._dims = self.expRange.shape[0]
        
        self.explorers = [Explorer(np.zeros(self._dims, dtype='float64') for i in range(numExplorers))]
        [self.drop_explorer(e) for e in self.explorers]
        
        self._bestExplorer = #TODO this will store the best sensor value as the explorers are stepped.
        
        self.rd = {}
        
        self._build_solution_space()
        
    def _experiential_model(self):
        
    def _certainty(self):
        #TODO add tensorflow elements to a graph to calculate the certainty value
        pass
    def _gratification_term(self):
        #TODO add tensorflow elements to a graph to calculate the gterm value
        pass
    def _sensor(self):
        #TODO add tensorflow elements to a graph to calculate the sensory value.
        pass
    def _full_pointValue(self):
        #TODO add pointValue elements to a graph to calculate the full pointValue.
        #       this will be differentiated with respect to the various control features.
        pass
    def _build_solution_space(self):
        self._certainty()
        self._gratification_term()
        self._sensor()
        self._full_pointValue()
        
    def drop_explorer(self, explorer):
        #TODO make more intelligent decisions about where to drop new explorers.
        drop = np.random.random_sample(size=self._dims)
        #multiply the drop by the range, the difference between the range limits.
        drop *= np.sum((self._expRange*np.array([[1,-1]])), axis=1)
        #add the bottom of the range to the drop value
        drop += self._expRange[:,0]
        #redrop the explorer
        explorer.redrop(drop)
        
    def step_explorer(self):
        #TODO after the explorers take some steps, 
        pass
