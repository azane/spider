import pymunk
import numpy
import math
from pymunk import Vec2d

from spider_nodes import node_dict
lib_nodes = node_dict() #get dictionary of node classes.

#temporary test lists for spider physiology #just using lists here because we do it once per runtime, and it's not intensive enough to warrant numpy
#TODO in the future, the spider physiology model will be gathered from files.
testBones = [
                        #[index, (relX, relY)|index, degreeRange]. #FIXME TODO bones attaching to bones at their endpoint is currently unimplemented.
                        #   this bone segment will go from index (-1 is the center), to the endpoint relative to the starting point,
                        #       for which it may eventually be allowed to be the endpoint of a previously indexed segment.
                        #indices must be less than its own.
                        #limitations on skeletal structures come only from muscles, there are no hard-coded angular constraints.
                        [-1, [30, 50]],
                        [0, [30, -60]],
                        [1, [-25, -30]],
                        [-1, [-30, 50]],
                        [3, [-30, -60]],
                        [4, [25, -30]]
            ]

#NOTE: even though muscles can be implemented as nodes, it makes sense to separate their specification like this to aid construction
#       eventually, the two might be integrated.
testMuscles = [
                        #[(sourceboneSegIndex, 0-1fractionDownSegment), (endBoneSeg, 0-1fractoinDownSeg), boolcontrollable] TODO add an elasticity modifier
                        #   the starting resting length will be derived from the bone structure with the bones resting in the middle.
                        #   resting length limits need to be proportional to the starting length. TODO this should be added as a modifier?
                        #indices cannot exceed those offered by the bonesArray
                        #the third value, a bool, determines whether or not this muscle has data collected, i.e. is stored in object storage in addition to drawing.
                        #NOTE that unless a designer wants to specifically constrain a body, all muscles should be controllable.
                        #       the brain will sort it out from there as far as what is helpful to control.
                        [[0, 0.95], [3, 0.95], False],
                        [[0, 0.35], [1, 0.95], False],
                        [[0, 0.25], [2, 0.85], True],
                        [[3, 0.35], [4, 0.95], False],
                        [[3, 0.25], [5, 0.85], False],
                        
                        [[0, 0.50], [1, 0.35], False],
                        [[3, 0.50], [4, 0.35], False]
                ]

#the below defines various nodes that a creature could have. these could be considered "organs".
#FIXME figure out where to put the node attachments.
        #in the bone gen loop? what if we make this a dict? and each bone end can have no more than one node attached?
        #like this testNodes = { -1: "balance", 0:"mouth", 1:"digestor" }
        #that would fix potential visualization problems with multiple nodes attached too close together.
#TODO node specifications should allow nodes to attach to other nodes, not just bones.
#FIXME node specs should allow multiple anchor points.
testNodes = [
                #[bone index at the end of which the node will be, "the type", {kwargs}] #FIXME kwargs should probably be generalized a little more?
                [0, "balance", {}], #the balance node
                [0, "deltax", {"mavgPeriod":450, "mavgPoints":225}] #the velocity tracking node
            ]

class SpiderPhysiology(object):
    
    """This class builds modularly builds the musculoskeletal structure of the spider.
        Then, it borrows from the nodes file to modularly add functionality."""
    
    def __init__(self, x, y, testBool, bonesArray=None, musclesArray=None, nodesArray=None):
        object.__init__(self)
        
        #get the spider's center origin location.
        self.x = x
        self.y = y
        
        
        
        #NOTE: muscles are nodes.
        #this list contains all the node objects that are a part of the spider.
        self.nodes = []
        
        #the 'dp_' prefix stands for 'draw' and 'physics'. these are lists of pymunk objects,
        #       not node objects, that the nodes return for drawing and physics.
        #include testBool (as opposed to setting defaults) so that one doesn't accidentally end up using a test skeletal array.
        #FIXME I think the physics and drawing reference lists need to be seperate. we don't need to draw all the joints, por ejemplo,
                #but it is needed that they are added to the space and calculated into the physics environment.
        if testBool:
            self.dp_bones, self.dp_nodes = self.genAnatomy(testBones, testMuscles, testNodes)
        else:
            if not bonesArray or not musclesArray or not nodesArray:
                raise ValueError("testBool is False, so bonesArray, musclesArray and nodesArray must be defined.")
            self.dp_bones, self.dp_nodes = self.genAnatomy(bonesArray, musclesArray, nodesArray)
        
    def draw_these(self):
        
        l = []
        
        l.extend(self.dp_bones)
        l.extend(self.dp_nodes)
        
        return l
        
    def apply_to_space(self, space):
        #add pymunk objects to the physics space.
        
        space.add(*self.dp_bones)
        space.add(*self.dp_nodes)
        
    def genAnatomy(self, bonesArray, musclesArray, nodesArray):
        
        #FIXME the nameage needs to change here. i've changed self.muscles and self.nodes to refer to the node objects referenceable for information.
        
        #FIXME clean this up. maybe change name to "gen musculoskeletal"
        
        segs = [] #list to store start/endpoints of bones [ 
                    #                                           0:{
                    #                                               {'sX':sX, 'sY':sY, 'eX':eX, 'eY':eY, 'shape':shape, 'body':body, 'l':l}
                    #                                           }
                    #                                        ]
        dp_bones = []
        
        #FIXME this will need to be different for each spider. it prevents self-collision.
        #       but maybe we don't want to prevent self-collision of some parts?
        #      anyway, it will need to be incremented by the worlds file.
        shapeGroup = 1
        
        #draw center circle
        cMass = 1
        cMoment = pymunk.moment_for_circle(cMass, 0, 5, (0,0))
        cBody = pymunk.Body(cMass, cMoment)
        cBody.position = (self.x, self.y)
        cShape = pymunk.Circle(cBody, 5, (0,0))
        cShape.friction = 0
        cShape.group = shapeGroup
        
        #start list with the center circle.
        dp_bones.extend([cBody, cShape])
        
        for i, b in enumerate(bonesArray):
            
            #get startpoints
            if b[0] == -1:
                #if this bone comes from the origin, set to 0 (origin)
                sX = 0
                sY = 0
            else:
                #FIXME error check for b[0] having been specified already.
                sX = segs[b[0]]['eX'] #get endpoints of parent segment, save as starting points.
                sY = segs[b[0]]['eY']
            #get endpoints
            eX = sX + b[1][0]
            eY = sY + b[1][1]
            
            #get length of bone
            l = math.sqrt((eX-sX)**2+(eY-sY)**2)
            points = [(0,0), (0,1), (l,0), (l,-1)] #relative to sX sY
            
            #set mass relative to bone length
            mass = l/10
            
            #print "points: " + str(points)
            #print "mass: " + str(mass)
            #print "body.position: " + str((sX+self.x, sY+self.y)) 
            
            moment = pymunk.moment_for_poly(mass, points, (0,0))
            body = pymunk.Body(mass, moment) #set body
            
            body.angle = numpy.arctan2(sY-eY, sX-eX) + 3.14#* 180 / math.pi
            #print "body.angle: " + str(body.angle)
            body.position = Vec2d(sX + self.x, sY + self.y) #set the body position relative to the center plus the center.
            
            shape = pymunk.Poly(body, points, (0,0))
            shape.group = shapeGroup
            shape.friction = 0.8
            
            if b[0] == -1:
                joint = pymunk.PivotJoint(cBody, body, (0,0), (0,0))
            else:
                joint = pymunk.PivotJoint(segs[b[0]]['body'], body, (segs[b[0]]['l'],0), (0,0))
                
            dp_bones.extend((body, shape, joint))
            
            #store start/end point info
            segs.append({'sX':sX, 'sY':sY, 'eX':eX, 'eY':eY, 'shape':shape, 'body':body, 'l':l})
            #print "segs: " + str(segs)
            
       
        dp_nodes = [] #list of nodes #remember muscles are nodes.
        for i, m in enumerate(musclesArray):
            
            fBI = m[0][0] #first bone index
            fBRatio = m[0][1] #ratio along first bone
            
            sBI = m[1][0] #second bone index
            sBRatio = m[1][1] #ratio along second bone
            
            stiffness = 2000 #m[2] #this is how badly it wants to return to its resting length
            damping = 150 #m[2] #this is the inverse of the rate the spring length changes.
            
            #calculate rest length from skeletal structure.
            #FIXME write a function for this normalization rest length point to point business.
            #get normalized vector
            #FIXME just rewrite the bones loop to save the actual vectors. doof.
            f_sVec = Vec2d(segs[fBI]['sX'], segs[fBI]['sY'])
            f_eVec = Vec2d(segs[fBI]['eX'], segs[fBI]['eY'])
            f_v = f_eVec - f_sVec
            f_nVec = f_v.normalized()
            f_point = f_sVec + (f_nVec*fBRatio*segs[fBI]['l'])
            #print "f_point: " + str(f_point.x) + ", " + str(f_point.y)
            
            s_sVec = Vec2d(segs[sBI]['sX'], segs[sBI]['sY'])
            s_eVec = Vec2d(segs[sBI]['eX'], segs[sBI]['eY'])
            s_v = s_eVec - s_sVec
            s_nVec = s_v.normalized()
            s_point = s_sVec + (s_nVec*sBRatio*segs[sBI]['l'])
            
            restingLength = f_point.get_distance(s_point)
            #print "restingLength: " + str(restingLength)
            #print "f_point: " + str(f_point)
            #print "s_point: " + str(s_point)
            
            muscle = lib_nodes['muscle'](
                                            anchorBodies=[segs[fBI]['body'], segs[sBI]['body']],
                                            anchorPoints=[(fBRatio*segs[fBI]['l'], 0), (sBRatio*segs[sBI]['l'], 0)],
                                            shapeGroup=shapeGroup,
                                            restLength=restingLength,
                                            stiffness=stiffness,
                                            damping=damping,
                                            report=m[2]  # set whether this muscle will be considered in data collection.
                                        )
            
            self.nodes.append(muscle) #append to node object list.
            
            mels = muscle.get_node_elements()
            assert type(mels)==list, "Node.get_node_elements must return a list."
            dp_nodes.extend(mels) #extend elements for drawing.
        
        for i, n in enumerate(nodesArray):
            
            if n[0] == -1:
                connBody = cBody
                connAnchor = (0,0)
            else:
                connBody = segs[n[0]]['body']
                connAnchor = (segs[n[0]]['l'],0)
                
            nodeClass = lib_nodes[n[1]]
            
            #pass the 3rd element of n, n[2] and expand it to kwargs. it is a dictionary containing node specific arguments
            #FIXME until node specs allow multiple anchors, do this.
            connBodies = [connBody] #TEMP
            connAnchors = [connAnchor] #TEMP
            nodeObject = nodeClass(connBodies, connAnchors, shapeGroup, **n[2])
            
            nels = nodeObject.get_node_elements()
            assert type(nels)==list, "Node.get_node_elements must return a list."
            dp_nodes.extend(nels)
            
            self.nodes.append(nodeObject)
        
        return dp_bones, dp_nodes
            