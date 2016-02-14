import pymunk as pymunk
from pymunk import Vec2d
import numpy as np
import math

"""This file returns a dictionary of classes that return pymunk configurations.
        Each class inherits from the main node class that requires a body and an anchor point with which to attach.
            nodes are not joints (though they can contain joints), and only have one attachment point.
            though, i suppose this means that nodes are modular themselves...because it would not be difficult to have a node attached to a node.
            though, i think the best way to provide information will be a first level access dictionary, so the first node would need to gather info
                from its children nodes.
        The nodes will also have various information accessible.
            This information may be sensory, for the brain of the body to which the node belongs.
            
            And/or, this information may result in events in the world, and can have an effect on other elements.
            FIXME as of 20151013, i'm not sure of the best way to have nodes interact with the world. so far, i only have a balance node and a velocity node,
                but those only hold information for use by the spider_brain
                maybe have the nodes handle the interactions?
                or have an interactions module reconcile node information with everything else?
                    then we'll have node info in multiple places...which is unnecessarily redundant.
                """
                
"""

Question: Should these nodes be built to return as much information as possible...or something like that, and that the brain
    should deal with parsing out what it deems useful?
    Or should the nodes (node writers) take some responsibility in only returning what will prove useful to the brain?

"""

#TODO part of a flexible, mutateable node model might be to have sigmoidal or gaussian settings for randomly generated defaults.
#           then, if the physiology mutates to take advantage of that particular node initialization argument, good!
#               but, if not, the values are still generated over the correct range.

#FIXME now that inheritance from pymunk classes has been removed, make these classes use the 'super' initialization functionality.

class BaseNode(object):
    """The base class for all nodes, including muscles.
    When initializing:
        'anchorBodies' is a list of the bodies to which the node will be anchored. This is usually a list of one element.
        'anchorPoints' is a list of the points, relative to the body of the same index in anchorBodies, of attachment to that body.
                        This is usually a list of one element.
        'shapeGroup' . shapes in the same non-zero group do not collide. in other words, a non-self-colliding node should have a non-zero shape group.
                        and all shapes of the built node should use self.shapeGroup
        'numAnchors' is the number of anchors. if this is passed, then the length of the anchor lists are verified.
    """
    
    #FIXME this class used the spi_ prefix to prevent namespace conflicts with doubly inherited pymunk classes.
    #       but, that double inheritance has been removed, so we don't need the prefixes anymore.
    
    def __init__(self, anchorBodies, anchorPoints, shapeGroup, numAnchors=None):
        object.__init__(self)
        
        assert type(anchorBodies) is list
        assert type(anchorPoints) is list
        if numAnchors is not None:
            assert type(numAnchors) is int
            assert len(anchorBodies) == numAnchors, "There should be " + str(numAnchors) + " anchorBody(ies) in the list."
            assert len(anchorPoints) == numAnchors, "There should be " + str(numAnchors) +  " anchorPoint(s) in the list."
        
        self.anchorBodies = anchorBodies
        self.anchorPoints = anchorPoints
        self.shapeGroup = shapeGroup
        
        #this updates the initial coordinates of the node's anchor points in self.worldXYs. unless added in child classes, this is the only call to this method.
        #       i.e. by default, self.worldXYs only holds the values when the node was initialized.
        self._spi_update_worldXYs()
        
        #build node
        self._spi_node_elements, self._spi_controlSize = self._build_node()
        assert type(self._spi_node_elements) is list, "_build_node must return a list as the first return value."
        assert type(self._spi_controlSize) is int, "_build_node must return an int as the second return value."
        
        #initialize control feature array size.
        #   control size must be defined during construction.
        #       this size is verified on every set and retrieve to guarantee that it does not change after construction.
        #   environmental and sensory should be set in step, but because the spider only reads these, the spider can just infer the size on retrieval.
        self._spi_data = {
                            'environmental': np.zeros(0),
                            'control': np.zeros(self._spi_controlSize),
                            'sensory': np.zeros(0)
                        }
        
    def _spi_update_worldXYs(self):
        """Return worldX and worldY of the anchorBodies and anchor points..
        Do not overwrite this method.
        """
        
        self.worldXYs = [ab.local_to_world(self.anchorPoints[i]) for i, ab in enumerate(self.anchorBodies)]
    
    def spi_get_info(self):
        """Return the data dictionary.
        Do not overwrite this method.
        """
        assert self._spi_data['control'].shape == (self._spi_controlSize,), "The control array does not match the size defined during construction."
        
        return self._spi_data
    
    def spi_set_control_features(self, control_array):
        """Set the passed control_array to data storage.
        Do not overwrite this method.
        """
        assert type(control_array) is np.ndarray
        assert control_array.shape == (self._spi_controlSize,), "The control array does not match the size defined during construction."
        #dtype assertions?
        
        self._spi_data['control'] = control_array
        
    
    def spi_step_and_get_info(self, dt):
        """Steps node to update data, and returns it in one call.
        Convenience method.
        Do not overwrite this method.
        """
        self.step(dt)
        return self.spi_get_info()
    
    def spi_node_elements(self):
        """Return a list of pymunk stuff to be drawn and physics...ized
        """
        return self._spi_node_elements
        
    #---<Overwrite These!>---
    def _build_node(self):
        """This method should be overwritten to
             1. return a list of individual node elements
             2. return the size of the control feature array
        Overwrite this method.
        """
        raise NotImplementedError("'_build_node' must be implemented in child classes."+\
                                    " It must build and return a list of pymunk elements for drawing and physics,"+\
                                        " and an int defining the size control feature array.")
    
    def step(self, dt):
        """This method should be overwritten to
            1. update _spi_data, if needed
            2. update node with incoming control features
        'dt' is delta time. this must be passed so that the node can keep track of rates of change.
        Overwrite this method.
        """
        raise NotImplementedError("step must be implemented in child classes. It should incorporate set control features and update _spi_data.")
    #---</Overwrite These!>---

class DeltaX(BaseNode):
    """The goal of this node is merely to track a moving average of DeltaX over a period.
    """
    def __init__(self, anchorBodies, anchorPoints, shapeGroup, mavgPeriod=25, mavgPoints=100):
        BaseNode.__init__(self, anchorBodies, anchorPoints, shapeGroup, numAnchors=1)
        
        self.points = np.zeros(mavgPoints)
        
        self.dtAgg = 0
        
        self.takePointAt = mavgPeriod/mavgPoints #take mavgPoints over mavgPeriod, mavgPoints in mavgPeriod.
        
        self.avgWeights = None #TODO generate a np array across an inverted log() for weight
        
        self.lastX = self.worldXYs[0][0] #init the lastX value to the initial worldX value of the only body.
        
    
    def _build_node(self):
        #this node has nothing that needs drawn or is involved in physics. and it has no control features.
        return [], 0
        
    def step(self, dt):
        """Calculates moving average and sets updates data dictionary.
        """
        self.dtAgg = self.dtAgg + dt #increment counter.
        
        #if the counter is less than the target, don't do anything.
        if self.dtAgg < self.takePointAt:
            return
        
        np.roll(self.points, 1) #roll array for new entry
        
        self._spi_update_worldXYs() #update global coordinates
        
        #get deltax by subtracting the last x from the current x of the only body
        #FIXME shouldn't this index be 0?
        self.points[1] = self.worldXYs[0][0] - self.lastX
        
        self.lastX = self.worldXYs[0][0] #update lastX to this worldX for next time.
        
        #average over the points, weighting with avgWeights
        f = np.average(self.points, weights=self.avgWeights)
        
        self._spi_data['sensory'] = np.array(f)
    
class SpiMuscle(BaseNode):
    """Define the muscle node.
    """
    def __init__(self, anchorBodies, anchorPoints, shapeGroup, restLength, stiffness, damping):
        BaseNode.__init__(self, anchorBodies, anchorPoints, shapeGroup, numAnchors=2)
        
        self.originalLength = restLength
    
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
        self.muscle = pymunk.DampedSpring(a=anchorBodies[0], b=anchorBodies[1],
                                            anchr1=anchorPoints[0], anchr2=anchorPoints[1],
                                            restLength=restLength, stiffness=stiffness, damping=damping)
        
        return [self.muscle], 1
        
    def step(self, dt):
        """Retrieve control features and update the data dictionary.
        """
        
        #set the environmental array to the length of the muscle.
        self._spi_data['environmental'] = np.array(self._get_length())
        
        #set to the first/only value in the control array.
        self.muscle.restLength = self._spi_data['control'][0]
        

class BalanceNode(BaseNode):
    def __init__(self, anchorBodies, anchorPoints, shapeGroup):
        BaseNode.__init__(self, anchorBodies, anchorPoints, shapeGroup, numAnchors=1)
        
        #store original anchorBody angle.
        self.origAnchorAngle = self.anchorBodies[0].angle
    
    def _get_balance(self):
        a = (self.segBody.angle - (self.anchorBodies[0].angle - self.origAnchorAngle))
        
        fa = np.arcsin(np.sin(a)) #i don't really know how it works, but this is a succinct way to get the range between -pi and pi. yay trig.
        #print fa
        return fa
    
    def _build_node(self):
        
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
        
        #return element list and control array size.
        return [self.segBody, segShape, segJoint], 0
    
    def step(self, dt):
        current = self._get_balance()
        
        self.deltaOverDT = (current - self.last)/dt
        
        self.last = current
        
        self._spi_data['environmental'] = np.array([current, self.deltaOverDT])


def node_dict():
    return {"balance": BalanceNode, "deltax": DeltaX}