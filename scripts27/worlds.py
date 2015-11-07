"""This file contains worlds. Each world inherits the pymunk.Space class, and is used as a space in the main.py file of this project.
    Each world has three methods, __init__() in which world-wide settings are selected, environment(), in which the non-agential elements are added to the space
        and agents(), in which all agents are added to the space.
        just for safety, methods not desiring to overwrite are prefixed with 'spi_', for the spider project."""

import pymunk
from pymunk import Vec2d
from pyglet.window import key, mouse

from spider_physiology import SpiderPhysiology
from spider_brain import SpiderBrain


##FIXME write a base class for worlds...it would need, some form of everything implemented below.

##FIXME make self.spi_spider = SpiderBrain(), not physiology, the physiology belongs to the brain.

class ConveyorTest(pymunk.Space):
    def __init__(self, spi_keys, spi_wh, spi_ww):
        
        pymunk.Space.__init__(self)
        
        #world-wide settings
        self.gravity = Vec2d(0.,-900)
        
        #FIXME clunky: get pyglet window height. there's gotta be a better way to do this. if only the pyglet window stored this attributes in a dictionary!
        #FIXME clunky: these are updated with the pyglet window being drawn, so resizing is okedoke.
        self.spi_wh = spi_wh
        self.spi_ww = spi_ww
        
        self.spi_keys = spi_keys
        
        self._spi_drawables = [] #that which is returned by the draw_these functions.
        self._spi_drawables.extend(self.spi_environment()) #add environment drawables
        self._spi_drawables.extend(self.spi_agents()) #add environment drawables
    
    def spi_timestep(self, dt):
        self.spi_brain.step(dt)
        
        """brainstorming
        
        here, we should pass dt to a reference object, a list/dict/object of some kind
        BaseNode (and all its children) will have access to this dt value
        that won't work, because how will the nodes know when to check the value of dt?
        
        is there a way to trigger all the nodes in the world without iterating through each one of them?
        what if only the nodes that need triggered get added to an iterator?
        
        what about a next(iterator) model?
        
        but actually, it makes sense for the SpiderBrain.step_data function to handle...well...stepping the data of its nodes.
        maybe we should rename this function to SpiderBrain.step, and it will call SpiderBrain._step_data and SpiderBrain._step_nodes
        both of which receive dt and can act however they wish on it.
        
        SpiderBrain._step_nodes then, will have access to the physiology, which has access to the nodes.
        SpiderBrain._step_nodes() will iterate the physiology's nodes and call BaseNode.step(dt) on each.
        BaseNode.step(dt) will simply pass and do nothing...but that can be overwritten in a child class.
            during the optimization phase of this program, SpiderBrain._step_nodes() should only iterate a list
            of nodes that want stepping. i.e. make another reference list in the physiology of only nodes that need stepping.
        
        """
    
    def spi_environment(self):
        wh = self.spi_wh
        ww = self.spi_ww
        
        l = []
        
        #put a conveyor on the bottom of the window. that's about it for this one.
        
        #conveyor base line, 0 friction.
        cbXOff = wh/10 #the x offset for the belt, from both sides
        cbYOff = 100 #the y offset for the belt, from the bottom.
        static_body = pymunk.Body()
        conveyor_base = pymunk.Segment(static_body, Vec2d(cbXOff,cbYOff), Vec2d(ww-cbXOff,cbYOff), 1)
        conveyor_base.friction = 30.0
        self.add(conveyor_base) #add to space for physics sim.
        l.extend([conveyor_base]) #add to list returned for drawing.
        
        return l#don't make belt.
        
        #conveyor "belt" with friction and mass (for inertia).
        cbHeight = 10
        cbWidth = (ww - (2*cbXOff))/2
        points = [(-cbWidth, -cbHeight), (-cbWidth, cbHeight), (cbWidth,cbHeight), (cbWidth, -cbHeight)]
        cbMass = 50
        cbMoment = pymunk.moment_for_poly(cbMass, points, (0,0))
        body = pymunk.Body(cbMass, cbMoment)
        body.position = Vec2d(ww/2, cbYOff + cbHeight)
        shape = pymunk.Poly(body, points, (0,0))
        shape.friction = 3.0
        self.add(body,shape) #add to space for physics sim
        l.extend([body, shape]) #add to list returned for drawing.
        
        return l
        
        
    def spi_draw_these(self):
        
        return self._spi_drawables
    
    def spi_agents(self):
        
        #for multi-agent worlds, this needs to return a list of agents, not drawables.
        
        l = []
        
        self.spi_spider = SpiderPhysiology(self.spi_ww/2, (self.spi_wh/20)*9, True)
        self.spi_spider.apply_to_space(self) #apply physics element to the space.
        
        l.extend(self.spi_spider.draw_these()) #pass draw
        
        self.spi_brain = SpiderBrain(self.spi_spider)
        
        return l
        
    def spi_control(self):
        m2 = self.spi_spider.muscles[2]
        #m1 = self.spi_spider.muscles[1]
        
        """if self.spi_keys[key.D]:
            m1.rest_length = m1.spi_originalLength * 1
        elif self.spi_keys[key.F]:
            m1.rest_length = m1.spi_originalLength * .75
        elif self.spi_keys[key.S]:
            m1.rest_length = m1.spi_originalLength * 1.25"""
            
        if self.spi_keys[key.K]:
            m2.rest_length = m2.spi_originalLength * 1
        elif self.spi_keys[key.J]:
            m2.rest_length = m2.spi_originalLength * .75
        elif self.spi_keys[key.L]:
            m2.rest_length = m2.spi_originalLength * 1.25
            
        if self.spi_keys[key.SPACE]:
            #self.spi_brain.series_to_csv(dest="data/foo.csv",columns=[0,1])
            self.spi_brain.data_to_csv(dest="data/full_with_n.csv")
            