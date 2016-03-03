import pymunk as pymunk
#from pymunk import Vec2d
import numpy as np

"""This file returns a dictionary of classes that return pymunk configurations.
        Each class inherits from the main node class that requires a list of bodies and anchor points to attach to.
    Nodes are only responsible for holding and dealing with control, environmental, and sensory feature data for the current timestep.
"""

#Question: Should these nodes be built to return as much information as possible...or something like that, and that the brain
#    should deal with parsing out what it deems useful?
#    Or should the nodes (node writers) take some responsibility in only returning what will prove useful to the brain?


#Thought: part of a flexible, mutateable node model might be to have sigmoidal or gaussian settings for randomly generated defaults.
#           then, if the physiology mutates to take advantage of that particular node initialization argument, good!
#               but, if not, the values are still generated over the correct range.

#FIXME 18691019djdks Nodes should not bother returning their control features. ExplorersHQ takes care of retaining those values.
#                       if the actual control feature might differ from the explorersHQ setting, then it should be returned as an environmental feature.

class BaseNode(object):
    """The base class for all nodes, including muscles.
    When initializing:
        'anchorBodies' is a list of the bodies to which the node will be anchored. This is usually a list of one element.
        'anchorPoints' is a list of the points, relative to the body of the same index in anchorBodies, of attachment to that body.
                        This is usually a list of one element.
        'shapeGroup' . shapes in the same non-zero group do not collide. in other words, a non-self-colliding node should have a non-zero shape group.
                        and all shapes of the built node should use self.shapeGroup
        'numAnchors' is the number of anchors. if this is passed, then the length of the anchor lists are verified.
        'report' is a bool that determines if the node should report any data. some nodes might only be used for constraint, for example.
    Variable Naming
        '_cts_data' is the current time step data, c.t.s. data.
    """
    
    def __init__(self, anchorBodies, anchorPoints, shapeGroup, numAnchors=None, report=True, **kwargs):
        super(BaseNode, self).__init__(**kwargs)  # FIXME edb18290hfhd . in this case, this would not be given kwargs.
        
        assert type(anchorBodies) is list
        assert type(anchorPoints) is list
        if numAnchors is not None:
            assert type(numAnchors) is int
            assert len(anchorBodies) == numAnchors, "There should be " + str(numAnchors) + " anchorBody(ies) in the list."
            assert len(anchorPoints) == numAnchors, "There should be " + str(numAnchors) +  " anchorPoint(s) in the list."
        
        self.anchorBodies = anchorBodies
        self.anchorPoints = anchorPoints
        self.shapeGroup = shapeGroup
        #mangle to prevent changing after initialization.
        self.__report = report
        
        #this updates the initial coordinates of the node's anchor points in self.worldXYs. unless added in child classes, this is the only call to this method.
        #       i.e. by default, self.worldXYs only holds the values when the node was initialized.
        self._update_worldXYs()
        
        #build node
        #   mangle the cts_data sizes to prevent post-initialization alterations.
        #FIXME TODO edb18290hfhd what about passing **kwargs to _build_node() so that class attributes don't have to be defined above the call to super init?
        #               and won't have to be defined as attributes at all?
        self._node_elements, controlDefault, environmentalDefault, sensoryDefault = self._build_node()
        assert type(self._node_elements) is list, "_build_node must return a list as the first return value."
        
        #initialize control feature array size.
        #   control size must be defined during construction.
        #       this size is verified on every set and retrieve to guarantee that it does not change after construction.
        #   environmental and sensory should be set in step, but because the spider only reads these, the spider can just infer the size on retrieval.
        try:
            self._cts_data = {
                                'control': np.array(controlDefault, dtype=np.dtype('float64')),
                                'environmental': np.array(environmentalDefault, dtype=np.dtype('float64')),
                                'sensory': np.array(sensoryDefault, dtype=np.dtype('float64'))
                            }
        except ValueError:
            raise ValueError("Defaults returned from _build_node must be array_like and convertible to float.")
        
        #set mangled size values to sizes from _build_node, this will ensure that step is consistent with what was defined in self._build_node
        self.__controlSize = self._cts_data['control'].size
        self.__environmentalSize = self._cts_data['environmental'].size
        self.__sensorySize = self._cts_data['sensory'].size
        
        self._verify_cts_data()
        
    def _update_worldXYs(self):
        """Return worldX and worldY of the anchorBodies and anchor points..
        Do not overwrite this method.
        """
        
        self.worldXYs = [ab.local_to_world(self.anchorPoints[i]) for i, ab in enumerate(self.anchorBodies)]
    
    def _verify_cts_data(self):
        
        assert type(self._cts_data['control']) is np.ndarray, "The control array must be a numpy ndarray."
        assert type(self._cts_data['environmental']) is np.ndarray, "The environmental array must be a numpy ndarray."
        assert type(self._cts_data['sensory']) is np.ndarray, "The sensory array must be a numpy ndarray."
        
        assert self._cts_data['control'].shape == (self.__controlSize,), "The control array does not match the size defined during construction."
        assert self._cts_data['environmental'].shape == (self.__environmentalSize,), "The control array does not match the size defined during construction."
        assert self._cts_data['sensory'].shape == (self.__sensorySize,), "The control array does not match the size defined during construction."
        
        #FIXME would it better to do this? or just force a flatten?
        assert self._cts_data['control'].ndim == 1, "The control array must be 1 dimensional."
        assert self._cts_data['environmental'].ndim == 1, "The environmental array must be 1 dimensional."
        assert self._cts_data['sensory'].ndim == 1, "The sensory array must be 1 dimensional."
        
        #this enables faster processing by the brain.
        assert self._cts_data['control'].dtype is np.dtype('float64'), "The control array dtype must be float64"
        assert self._cts_data['environmental'].dtype is np.dtype('float64'), "The environmental array dtype must be float64"
        assert self._cts_data['sensory'].dtype is np.dtype('float64'), "The sensory array dtype must be float64"
        
    def get_data(self):
        """Return the data dictionary.
        Do not overwrite this method.
        """
        
        #FIXME there must be a better way to check these than checking every time the data is accessed?
        #verify stuff on the outgoing end to catch mistakes made in child classes.
        self._verify_cts_data()
        
        if self.__report:
            return self._cts_data
        else:
            #return an empty data dictionary if report is false.
            return {
                        'control': np.zeros(0),
                        'environmental': np.zeros(0),
                        'sensory': np.zeros(0)
                    }
                    
    def set_control_features(self, control_array):
        """Set the passed control_array to the data dictionary.
        Do not overwrite this method.
        """
        #TODO write set methods for environmental and sensory? but just for private use? these would serve to verify the build function.
        
        #verify incoming
        try:
            control_array = np.array(control_array, dtype=np.dtype('float64'))
        except ValueError:
            raise ValueError("The input control array must be array like and convertible to float64.")
        assert control_array.shape == (self.__controlSize,), "The input control array does not match the size defined during construction."
        
        #in place mods
        self._cts_data['control'] = control_array
        
    
    def step_and_get_data(self, dt):
        """Steps node to update data, and returns it in one call.
        Convenience method.
        Do not overwrite this method.
        """
        self.step(dt)
        return self.get_data()
    
    def get_node_elements(self):
        """Return a list of pymunk stuff to be drawn and physics...ized
        """
        return self._node_elements
        
    #---<Overwrite These!>---
    def _build_node(self):
        """This method should be overwritten to, in this order,
             1. return a list of individual node elements
             #      the below defaults must be array_like
             2. return the default of the control feature array
             3. return the default of the environmental feature array
             4. return the default of the sensory feature array
        This requirement is to force new nodes to be very explicit about the sizes of their data arrays.
        Overwrite this method.
        """
        raise NotImplementedError("'_build_node' must be implemented in child classes."+\
                                    " It must build and return a list of pymunk elements for drawing and physics,"+\
                                        " and defaults defining the control, environmental, and sensory features, in that order.")
        #e.g.
        #return [pymunkThingy], [controlDefault], [environmentalDefault], [sensoryDefault]
        
    def step(self, dt):
        """This method should be overwritten to
            1. update _cts_data, if needed
            2. update node with incoming control features
        'dt' is delta time. this must be passed so that the node can keep track of rates of change.
        Overwrite this method.
        """
        raise NotImplementedError("step must be implemented in child classes. It should incorporate set control features and update _cts_data.")
    #---</Overwrite These!>---

class DeltaX(BaseNode):
    """The goal of this node is merely to track a moving average of DeltaX over a period.
    """
    def __init__(self, anchorBodies, anchorPoints, shapeGroup, mavgPeriod=25, mavgPoints=100, **kwargs):
        super(DeltaX, self).__init__(anchorBodies, anchorPoints, shapeGroup, numAnchors=1, **kwargs)
        
        self.points = np.zeros(mavgPoints)
        
        self.dtAgg = 0
        
        self.takePointAt = mavgPeriod/mavgPoints #take mavgPoints over mavgPeriod, mavgPoints in mavgPeriod.
        
        self.avgWeights = None #TODO generate a np array across an inverted log() for weight
        
        self.lastX = self.worldXYs[0][0] #init the lastX value to the initial worldX value of the only body.
        
    
    def _build_node(self):
        #return no elements for dp. return the sensory feature default.
        return [], [], [], [0]
        
    def step(self, dt):
        """Calculates moving average and sets updates data dictionary.
        """
        self.dtAgg = self.dtAgg + dt #increment counter.
        
        #if the counter is less than the target, don't do anything.
        if self.dtAgg < self.takePointAt:
            return
        
        np.roll(self.points, 1) #roll array for new entry
        
        self._update_worldXYs() #update global coordinates
        
        #get deltax by subtracting the last x from the current x of the only body
        #FIXME shouldn't this index be 0?
        self.points[1] = self.worldXYs[0][0] - self.lastX
        
        self.lastX = self.worldXYs[0][0] #update lastX to this worldX for next time.
        
        #average over the points, weighting with avgWeights
        f = np.average(self.points, weights=self.avgWeights)
        
        #TEMP FIXME a temp fix for poor gmm performance on small values.
        f *= 100
        
        self._cts_data['sensory'] = np.array([f])
    
class SpiMuscle(BaseNode):
    """Define the muscle node.
    """
    def __init__(self, anchorBodies, anchorPoints, shapeGroup, restLength, stiffness, damping, **kwargs):
        
        #FIXME TODO find edb18290hfhd for more info.
        self.stiffness = stiffness
        self.damping = damping
        self.originalLength = restLength
        
        super(SpiMuscle, self).__init__(anchorBodies, anchorPoints, shapeGroup, numAnchors=2, **kwargs)
    
    def _get_length(self):
        """Return the current length of the muscle.
        """
        aWorld = self.muscle.a.local_to_world(self.muscle.anchr1)
        bWorld = self.muscle.b.local_to_world(self.muscle.anchr2)
        
        return np.sqrt(aWorld.get_dist_sqrd(bWorld))
    
    def _build_node(self):
        """Return a list of node elements to draw.
           Return the size of the conrol feature array.
           """
        self.muscle = pymunk.DampedSpring(a=self.anchorBodies[0], b=self.anchorBodies[1],
                                            anchr1=self.anchorPoints[0], anchr2=self.anchorPoints[1],
                                            #can use original length because _build only gets called on init.
                                            rest_length=self.originalLength, stiffness=self.stiffness, damping=self.damping)
        
        #return elements, and control and environmental defaults.
        return [self.muscle], [0], [0], []
        
    def step(self, dt):
        """Retrieve control features and update the data dictionary.
        """
        
        #set the environmental array to the length of the muscle.
        self._cts_data['environmental'] = np.array([self._get_length()])
        
        #set to the first/only value in the control array.
        #TODO gjdkd129287 should the step function/set control function scale values from a required range?
        #       as the spider will be using a tanh output, -1, 1
        #       so if the control features scaled off that range, the brain wouldn't have to scale it.
        #       but, i imagine that that requirement might limit flexibility down the road,
        #       and since the brain can/should be able to handle that kind of scaling...it should.
        #       although, it does make sense to have '0' be the middle ground?
        #       ...we could do both? have the brain be able to handle scaling, but have control features scale from (-1,1)?
        #       perhaps it does really make sense for this node in particular. but for others, maybe not. so both it is!?
        #   yes, because nodes are responsible for providing 'real world' limitations. the brain is responsible for discovering the futility of setting
        #       control features to values that won't make the node respond any more.
        #self.muscle.rest_length = (self._cts_data['control'][0] + 1)*self.originalLength
        olMultiplier = np.tanh(self._cts_data['control'][0])/3  # keep within -0.5 and 0.5
        olMultiplier += 1.  # add one so the multiplier is between 0.5 and 1.5
        self.muscle.rest_length = olMultiplier*self.originalLength
        

class BalanceNode(BaseNode):
    def __init__(self, anchorBodies, anchorPoints, shapeGroup):
        super(BalanceNode, self).__init__(anchorBodies, anchorPoints, shapeGroup, numAnchors=1)
    
    def _get_balance(self):
        a = (self.segBody.angle - (self.anchorBodies[0].angle - self.origAnchorAngle))
        
        fa = np.arcsin(np.sin(a)) #i don't really know how it works, but this is a succinct way to get the range between -pi and pi. yay trig.
        #print fa
        return fa
    
    def _build_node(self):
        
        #store original anchorBody angle.
        self.origAnchorAngle = self.anchorBodies[0].angle
        
        #create a "plumb" segment connected to the circle.
        segMass = 1
        a = (0,0)
        b = (0,-20)
        segMoment = pymunk.moment_for_segment(segMass, a, b)
        
        self.segBody = pymunk.Body(segMass, segMoment)
        segShape = pymunk.Segment(self.segBody, a, b, 1)
        segShape.group = self.shapeGroup
        #segShape.sensor = True #TODO get rid of this line if it doesn't throw errors being gone.
        self.segBody.position = self.worldXYs[0] #position at the node's world anchor point.
        
        #create a pivot joint from the circle to the segment
        segJoint = pymunk.PivotJoint(self.anchorBodies[0], self.segBody, self.anchorPoints[0], (0,0))
        
        self.last = self._get_balance() #the last balance.
        self.deltaOverDT = 0 #start slope at 0
        
        #return element list and environmental array default.
        return [self.segBody, segShape, segJoint], [], [0], []
    
    def step(self, dt):
        current = self._get_balance()
        
        self.deltaOverDT = (current - self.last)/dt
        
        self.last = current
        
        self._cts_data['environmental'] = np.array([current])#, self.deltaOverDT])


def node_dict():
    return {"balance": BalanceNode, "deltax": DeltaX, "muscle": SpiMuscle}