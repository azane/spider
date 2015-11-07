import pymunk
from pymunk import Vec2d
import numpy
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
            FIXME as of 20151013, i'm not sure of the best way to have nodes interact with the world. so far, i only have a balance node,
                and that only holds information for use by the spider_brain
                maybe have the nodes handle the interactions?
                or have an interactions module reconcile node information with everything else?
                    then we'll have node info in multiple places...which is unnecessarily redundant.
                """
                
"""

Question: Should these nodes be built to return as much information as possible...or something like that, and that the brain
    should deal with parsing out what it deems useful?
    Or should the nodes (node writers) take some responsibility in only returning what will prove useful to the brain?

"""

                
class BaseNode(object):
    def __init__(self, body, anchorPoint, shapeGroup, environment=True, sensor=False):
        object.__init__(self)
        
        """environment and sensor bools determine whether or not the node will be recorded, in data, as an environment variable and/or a sensor.
            where 'or' is naturally non-exclusive"""
        
        self.anchorBody = body #the body to which the node is connected
        self.anchorPoint = anchorPoint
        self.shapeGroup = shapeGroup
        
        #this updates the initial coordinate of the node center. worldX and worldY are not kept updated unless this is called.
        self._update_worldXY()
        
        self.environment = environment
        self.sensor = sensor
        
    def _update_worldXY(self):
        self.worldX, self.worldY = self.anchorBody.local_to_world(self.anchorPoint)
        
    def node_elements(self):
        raise NotImplementedError("'node_elements' must be implemented in child classes. It should return all pymunk bodies, shapes, constraints, etc. of the node.")
    def get_info(self):
        raise NotImplementedError("'get_info' must be implemented in child classes. It should return a list of the sensor's information.")
    def step(self, dt):
        pass #unimplemented this does nothing, but it's not considered an error.

#FIXME TODO this node needs some semblance of dt (which would be good for the BaseNode class to have anyway) 
#               AND the BaseNode class needs to be able to handle kwargs that will be passed as a dictionary
#               from spider_physiology upon creation. Although, if this kwargs feature was modified to take lists, it would make physiological
#               mutation much easier to implement.

#TODO part of a flexible, mutateable node model might be to have sigmoidal or gaussian outputs for a randomly generated default.
#           then, if the physiology mutates to take advantage of that particular node initialization argument, good!
#               but, if not, the values are still generated over the correct range.
class DeltaX(BaseNode):
    def __init__(self, body, anchorPoint, shapeGroup, mavgPeriod=25, mavgPoints=100, **kwargs):
        BaseNode.__init__(self, body, anchorPoint, shapeGroup, **kwargs)
        
        """The goal of this node is merely to track a moving average of DeltaX over a period defined by the physiology.
        """
        
        self.points = numpy.zeros(mavgPoints)
        
        self.dtAgg = 0
        
        self.takePointAt = mavgPeriod/mavgPoints #take mavgPoints over mavgPeriod, mavgPoints in mavgPeriod.
        
        self.avgWeights = None #TODO generate a numpy array across an inverted log() for weight
        
        self.lastX = self.worldX #init the lastX value to the initial worldX value.
        
    
    def node_elements(self):
        return [] #nothing to see here.
        
    def spi_get_info(self):
        
        f = numpy.average(self.points, weights=self.avgWeights)
        
        #print "deltaX avg: " + str(f)
        
        return [f]
        
    def step(self, dt):
        
        self.dtAgg = self.dtAgg + dt #increment counter.
        
        #if the counter is less than the target, don't do anything.
        if self.dtAgg < self.takePointAt:
            return
        
        numpy.roll(self.points, 1) #roll array for new entry
        
        self._update_worldXY() #update global coordinates (worldX)
        
        self.points[1] = self.worldX - self.lastX #store delta x
        
        self.lastX = self.worldX #update lastX to this worldX for next time.
        
    
#the DampedSpring spi class that provides a get_info bit.
class SpiMuscle(pymunk.DampedSpring):
    #TODO implement a control feature method.
    def __init__(self, *args, **kwargs):
        #TODO this class needs dt as well to calculate the slope.
        pymunk.DampedSpring.__init__(self, *args, **kwargs)
        
        #set for muscles.
        self.environment = True
        self.sensor = False
        
        self.last = self._spi_get_length() #for delta
        
        self.deltaOverDT = 0 #set delta to zilcho
    
    def _spi_get_length(self):
        aWorld = self.a.local_to_world(self.anchr1)
        bWorld = self.b.local_to_world(self.anchr2)
        
        return numpy.sqrt(aWorld.get_dist_sqrd(bWorld))
        
    def spi_get_info(self):
        
        return [self._spi_get_length()]
        #return [self.deltaOverDT]
    
    def step(self, dt):
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
        
        
    def node_elements(self):
        #FIXME maybe actually build the elements in here? otherwise sloppy.
        return self.elements
    
    def spi_get_info(self):
        #return [self.deltaOverDT]
        return [self._get_balance()]
    
    def _get_balance(self):
        a = (self.segBody.angle - (self.anchorBody.angle - self.origAnchorAngle))
        
        fa = numpy.arcsin(numpy.sin(a)) #i don't really know how it works, but this is a succinct way to get the range between -pi and pi
        #print fa
        return fa
    def step(self, dt):
        current = self._get_balance()
        
        self.deltaOverDT = (current - self.last)/dt
        
        self.last = current

#TODO implement a wrapper node for the muscles. this way it can be referenced in the same manner as Nodes. But, we'll need to inherit from pymunk.DampedSpring

def node_dict():
    return {"balance": BalanceNode, "deltax": DeltaX}