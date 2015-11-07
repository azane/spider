import numpy
import pandas
import matplotlib.pyplot as pyplot
import pylab as pylab

"""

remember that one of the main premises is that the series data is not stored perpetually, but rather converted to a (or many) polynomials.
    this allows for a very compact storage of complex interactions. as new information comes in, the models are updated and refined.
    acting on these functions is pheasable...as opposed to an enormous set of points and clusters. like...100*100 vs 100^100. O_o

"""

"""NOTE:

real quick cz i don't know where else to put it. we'll want to give the brain the ability to determine how affective control changing will be.
for example, if it's calculated that the right muscle is supporting a lot of weight, it will be compressed, and modifying the resting length of the left
    muscle will prove more affective. (and yes, i mean affective, not effective)

"""

class SpiderBrain(object):
    def __init__(self, physiology):
        object.__init__(self)
        
        self.physiology = physiology #a SpiderPhysiology object so the brain can access the muscles.
        
        #muscles are considered nodes. #FIXME just combine these in physiology.
        self.allNodes = []
        self.allNodes.extend(self.physiology.muscles)
        self.allNodes.extend(self.physiology.nodes)
        
        #the dimensions store information for creating
        
        ##All things set to None are shaped in self.__define_series()
        
        self.cts = None #the current time step, cts. rows by object, columns by info element.
        
        #2d over t, with self.cts flattened over each object's info elements.
        #FIXME maybe this should be an appendable list of numpy arrays (coords) that are all lumped together in data
        self.series = None #rows along t, columns are the flattened version of self.cts with nans removed.
        
        #FIXME will we eventually need a self.data for EACH sensor we want to cast over n? for this stage, just one....maybe multiple are pheasible.
        #2d as a list of coordinates.
        #this differes from series in that many coordinates are generated over self.lookback for each timestep of self.series.
        self.data = None #the array to which self.cts is appended after all information is gathered to self.ct for that timestep.
        self.dataStore = 7000 #this denotes the amount of coordinates to keep on hand, in memory. these are cleared out as they are processed,
                                # so it's possible that this may never fill up.
        
        #FIXME we might want to consider lookBack as a max, over which we cast the sensor over n, but at increasing intervals (log probs) as max is approached
        self.lookBack = 25 #this is how many points are generated over n at each moment.
        self.timeStep = 0.05 #the interval at which cts is updated.
        self.dtAgg = 0 #the counter checked against self.timeStep to increment series data.
        self.seriesSize = self.lookBack #this denotes how long data is stored in the time series. for graphing purposes, it may be valuable for this to exceed 
                                                #lookBack, but if visualization is not of concern, this should equal lookback, but never be less than lookback.
        #so, for each call to self.step_data, self.lookback points are generated, where the dt between each is self.timestep
        #further, the time series are dumped after self.lookback*self.timestep
        
        if self.seriesSize < self.lookBack: ValueError("seriesSize must be >= lookBack.") #should we put an error in here, or change to a +self.lookBack model?
        
        self.__define_series()
        
        
    def __define_series(self):
        
        #FIXME document this more/ensure it? physiologies must be ordered, i.e., they must return objects in the same order every time.
        
        total = 0
        pDatList = []
        
        for i, p in enumerate(self.allNodes):
            if p.environment or p.sensor: #only if data should be recorded
                pDatLen = len(p.spi_get_info())
                #print "pDatLen: " + str(pDatLen)
            
                pDatList.append(pDatLen)
            
                total = total + pDatLen
        
        #print str(i) + ", " + str(max(*pDatList))
        
        #set self.cts as a box array with nans. when the cts is updated, nans will be overwritten as required.
        #FIXME ensure that this is indexically ordered
            #i+1 cz i is a zero base index, not a count. shapes are defined as counts.
        self.cts = numpy.full((i+1, max(*pDatList)), numpy.nan) #this array's memory allotment will remain, but the numbers will shift around for each timestep.
        
        #print self.cts
        
        #FIXME ensure that this is indexically ordered.
        self.series = numpy.full((self.seriesSize, total), numpy.nan) #set size of series, each coordinate is the flattened version of self.cts without nans.
        #print "total: " + str(total)
        #print "self.series: "
        #print self.series
        #print
        
        #total+2 because of the additional n dimension, and the addition of the output sensor.
        #   note that the output sensor cast over n may be present as an input as well.
        self.data = numpy.full((self.dataStore, total+2), numpy.nan)
        #print "self.data: "
        #print self.data
        #print
    def step(self, dt):
        self._step_nodes(dt)
        self._step_data(dt)
    def _step_nodes(self, dt):
        #FIXME FIXME we might want to limit this iteration to only once every few dt, rather than every single one. it could limit FPS significantly.
        for n in self.allNodes:
            n.step(dt)
    def _step_data(self, dt):
        
        ##TODO create multiple output graphs for each sensor so defined so they can be maximized.
                #FIXME are multiple graphs needed? or can relationships be extrapolated from one and assigned to others?
                #       yesh. especially if we relate the data 1 by 1, and only use formulas as needed. for specific sensors.
        
        #check timestep.
        self.dtAgg = self.dtAgg + dt
        if self.dtAgg < self.timeStep:
            return
        #if it passes, reset dtAgg
        self.dtAgg = 0
        
        #The array modifications in this method shy away from replacing anything alotted in __define_series()
        
        sensorList = [] #the list of sensor indices
        
        #gather info from physiology and update the current time step.
        ctsIndex = -1 #counter for how far through the cts array we are. start at -1 cz index.
        for i, p in enumerate(self.allNodes):
            if p.environment or p.sensor:
                temp = numpy.asarray(p.spi_get_info(), dtype=numpy.float64)
                #for the relevant row, fill only as many columns as are returned, leaving the rest nan.
                self.cts[i-1][:numpy.ma.size(temp)] = temp
                ctsIndex = ctsIndex + 1
            
            if p.sensor:
                sensorList.append(ctsIndex) #this was updated in the last conditional, so no need for a +1
        
        #flatten and remove nans
        tempCurrent = self.cts.flatten() #flatten to coordinate shape
        tempCurrent = tempCurrent[~numpy.isnan(tempCurrent)] #remove nans
        
        #load this coordinate into the time series array.
            #roll series array
        self.series = numpy.roll(self.series, 1, axis=0) #FIXME verify that this doesn't reallocate memory.
            #replace the first (rolled, so the "last") element with the new data.
        self.series[0] = tempCurrent
        
        print "tempCurrent/self.series[0]:"
        print self.series[0]
        print
        
        
        #FIXME for now, just take the first element in the list, later, we might want to iterate over this to generate data for many sensors cast over n.
        sIndex = sensorList[0]
        
        
        print "self.data pre-roll: "
        print self.data
        print
        #roll self.data self.lookback times, prep for fill.
        self.data = numpy.roll(self.data, self.lookBack, axis=0) #FIXME verify that this doesn't reallocate memory.
        print "self.data post-roll: "
        print self.data
        print
        
        #dView = self.data[:self.lookBack, :] #get view of the newly rolled section of self.data
        
        #broadcast self.series over self.lookBack to dView, this will leave a space for n, as self.data rows are 1 longer than self.series rows.
        self.data[:self.lookBack, :-2] = self.series[:self.lookBack, :]
        print "self.data post-series-broadcast: "
        print self.data
        print
        
        #init an array with self.lookBack rows and fill it with the value of the sensor
        sensorColumn = numpy.full(self.lookBack, tempCurrent[sIndex])
        print "sensorColumn: "
        print sensorColumn
        print
        #broadcast this array onto the last element of the data coordinates, the sensor
        self.data[:self.lookBack, -1] = sensorColumn
        print "self.data post-sensorColumn broadcast: "
        print self.data
        print
        
        #for n: init an array with an indexical distribution over the range self.lookBack, from 0:(self.lookBack-1), 2d, with self.lookBack rows and 1 column.
        nRange = numpy.arange(self.lookBack) * (-1) #turn negative so that "back in time" is negative. not really mathematically important, except for consistency.
        print "nRange: "
        print nRange
        print
        #broadcast this array onto the last element in each row of dView
        self.data[:self.lookBack, -2] = nRange
        
        print "self.data post-nRange broadcast"
        print self.data
        print
        
        print
        print "--------------------------------------------------------------"
        print
        print
        
    def plot_time_series(self):
        
        """m1Series = self.series[:,[2]].flatten() #i think these are the right ones?
        m2Series = self.series[:,[4]].flatten() #?
        n1Series = self.series[:,[7]].flatten()"""
        
        
        testSeries = [1, 2, 3]
        #print "testSeries:"
        #print testSeries
        
        pyplot.plot(testSeries)
        pyplot.show()
        
    def series_to_csv(self, dest="foo.csv", columns=[]):
        
        #TODO change the defaults, and finish this method.
        
        #FIXME columns needs to select all the columns by default, but it keeps throwing "baddy synax" errors when it's defined apart from an arr-like.
        
        data = self.series[:,columns]
        
        numpy.savetxt(dest, data, fmt='%.3f', delimiter=",")
    
    def data_to_csv(self, dest="foo.csv"):
        
        numpy.savetxt(dest, self.data, fmt='%.3f', delimiter=",")
        
        
        