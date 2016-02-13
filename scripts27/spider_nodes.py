import pymunk
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


class BaseNode(object):
    """The base class for all nodes, including muscles.
    When initializing:
        'body' is the body to which the node will be attached.
        'anchorPoint' is the point, relative to the body, of attachment to that body.
        'shapeGroup' is ??
    """
    
    #FIXME this class uses the spi_ prefix to prevent namespace conflicts with doubly inherited pymunk classes.
    #       but, i think it's probably better to not have nodes inherit from pymunk classes, and instead just have those objects as node elements.
    #      this fix will require a rewrite of the SpiMuscle node.
    
    def __init__(self, body, anchorPoint, shapeGroup):
        object.__init__(self)
        
        self.anchorBody = body
        self.anchorPoint = anchorPoint
        self.shapeGroup = shapeGroup
        
        #this updates the initial coordinate of the node center. this is not called any other time, by default.
        self.worldX, self.worldY = self._spi_update_worldXY()
        
        #build node
        self._spi_node_elements, self._spi_controlSize = self._spi_build_node()
        assert type(self._spi_node_elements) is list, "_spi_build_node must return a list as the first return value."
        assert type(self._spi_controlSize) is int, "_spi_build_node must return an int as the second return value."
        
        #initialize control feature array size.
        self._spi_data = {
                            'environmental': None,
                            'control': np.zeros(self._spi_controlSize),
                            'sensory': None
                        }
        
    def _spi_update_worldXY(self):
        """Return worldX and worldY.
        Do not overwrite this method.
        """
        return self.anchorBody.local_to_world(self.anchorPoint)
    
    def spi_get_info(self):
        """Return the data dictionary after updating it.
        Do not overwrite this method.
        """
        return self._spi_data
    
    def spi_set_control_features(self, control_array):
        """Set the passed control_array to data storage.
        Do not overwrite this method.
        """
        assert type(control_array) is np.ndarray
        assert control_array.shape == self._spi_data['control'].shape
        #dtype assertions?
        
        self._spi_data['control'] = control_array
        
    
    def spi_step_and_get_info(self, dt):
        """Steps node to update data, and returns it in one call.
        Convenience method.
        Do not overwrite this method.
        """
        self.spi_step(dt)
        return self.spi_get_info()
    
    def spi_node_elements(self):
        """Return a list of pymunk stuff to be drawn and physics...ized
        """
        return self._spi_node_elements
        
    #---<Overwrite These!>---
    def _spi_build_node(self):
        """This method should be overwritten to
             1. return a list of individual node elements
             2. return the size of the control feature array
        Overwrite this method.
        """
        raise NotImplementedError("'_spi_build_node' must be implemented in child classes."+\
                                    " It must build and return a list of pymunk elements for drawing and physics,"+\
                                        " and an int defining the size control feature array.")
    
    def spi_step(self, dt):
        """This method should be overwritten to
            1. update _spi_data, if needed
            2. update node with incoming control features
        Overwrite this method.
        """
        raise NotImplementedError("spi_step must be implemented in child classes. It should incorporate set control features and update _spi_data.")
    #---</Overwrite These!>---

class DeltaX(BaseNode):
    """The goal of this node is merely to track a moving average of DeltaX over a period.
    """
    def __init__(self, body, anchorPoint, shapeGroup, mavgPeriod=25, mavgPoints=100, **kwargs):
        BaseNode.__init__(self, body, anchorPoint, shapeGroup, **kwargs)
        
        self.points = np.zeros(mavgPoints)
        
        self.dtAgg = 0
        
        self.takePointAt = mavgPeriod/mavgPoints #take mavgPoints over mavgPeriod, mavgPoints in mavgPeriod.
        
        self.avgWeights = None #TODO generate a np array across an inverted log() for weight
        
        self.lastX = self.worldX #init the lastX value to the initial worldX value.
        
    
    def spi_node_elements(self):
        return []  # nothing to see here.
        
    def spi_get_info(self):
        
        f = np.average(self.points, weights=self.avgWeights)
        
        #print "deltaX avg: " + str(f)
        
        return [f]
        
    def spi_step(self, dt):
        
        self.dtAgg = self.dtAgg + dt #increment counter.
        
        #if the counter is less than the target, don't do anything.
        if self.dtAgg < self.takePointAt:
            return
        
        np.roll(self.points, 1) #roll array for new entry
        
        self._update_worldXY() #update global coordinates (worldX)
        
        self.points[1] = self.worldX - self.lastX #store delta x
        
        self.lastX = self.worldX #update lastX to this worldX for next time.
        
    
class SpiMuscle(pymunk.DampedSpring, BaseNode):
    
    """this 'node' inherits from the pymunk DampedSpring and BaseNode.
    Muscles are grouped with other nodes for drawing, updating, and data collection."""
    def __init__(self, *args, **kwargs):
        
        #NOTE: pymunk.DampedSpring does not use super(), so we need to do it this way.
        if args:
            raise AttributeError("SpiMuscle only accepts key word arguments. Normal arguments an be defined as key words.")
        pymunk.DampedSpring.__init__(self, **kwargs)
        
        #<---BaseNode.__init__ Replacement>
        #don't call BaseNode.__init__, as the body and anchors are tracked by DampedSpring
        #   but we do need a few things.
        #set for muscles.
        #self.environment = True
        #self.sensor = False
        self.environment = environment
        self.sensor = sensor
        #<---BaseNode.__init__ Replacement>
        
        self.spi_originalLength = spi_originalLength
        
        self.last = self._spi_get_length() #for delta
        
        self.deltaOverDT = 0 #set delta to zilcho
    
    def _spi_get_length(self):
        aWorld = self.a.local_to_world(self.anchr1)
        bWorld = self.b.local_to_world(self.anchr2)
        
        return np.sqrt(aWorld.get_dist_sqrd(bWorld))
        
    def spi_get_info(self):
        
        return [self._spi_get_length()]
        #return [self._spi_get_length(), self.deltaOverDT] #to be consistent with a "max information" philosophy.
    
    def spi_node_elements(self):
        #return a list of node elements. the muscle uses itself, as it inherits from DampedSpring.
        return [self]
        
    def spi_step(self, dt):
        current = self._spi_get_length()
        
        self.deltaOverDT = (current - self.last)/dt
        
        self.last = current

class BalanceNode(BaseNode):
    def __init__(self, body, anchorPoint, shapeGroup, **kwargs):
        BaseNode.__init__(self, body, anchorPoint, shapeGroup, **kwargs)
        
        #store original anchorBody angle.
        self.origAnchorAngle = self.anchorBody.angle
        
        #create a "plumb" segment connected to the circle.
        segMass = 1
        a = (0,0)
        b = (0,-20)
        segMoment = pymunk.moment_for_segment(segMass, a, b)
        
        self.segBody = pymunk.Body(segMass, segMoment)
        segShape = pymunk.Segment(self.segBody, a, b, 1)
        segShape.group = self.shapeGroup
        segShape.sensor = True
        self.segBody.position = (self.worldX, self.worldY) #position at the node's world anchor point.
        
        #create a pivot joint from the circle to the segment
        segJoint = pymunk.PivotJoint(self.anchorBody, self.segBody, self.anchorPoint, (0,0))
        
        self.elements = [self.segBody, segShape, segJoint]#, self.cBody, cShape, cJoint, cRotaryLimit]
        
        self.last = self._get_balance() #the last balance.
        self.deltaOverDT = 0 #start slope at 0
        
        
    def spi_node_elements(self):
        #FIXME build the elements in here, not in __init__
        return self.elements
    
    def spi_get_info(self):
        #return [self.deltaOverDT]
        return [self._get_balance()]
    
    def _get_balance(self):
        a = (self.segBody.angle - (self.anchorBody.angle - self.origAnchorAngle))
        
        fa = np.arcsin(np.sin(a)) #i don't really know how it works, but this is a succinct way to get the range between -pi and pi. yay trig.
        #print fa
        return fa
    def spi_step(self, dt):
        current = self._get_balance()
        
        self.deltaOverDT = (current - self.last)/dt
        
        self.last = current

"""brainstorming

the control features' node affectation should happen in the code of those nodes.
    this makes the most sense as far as

"""


def node_dict():
    return {"balance": BalanceNode, "deltax": DeltaX}