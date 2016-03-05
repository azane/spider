import numpy as np
import os as os

class SpiderBrain(object):
    def __init__(self, physiology, explorerHQ):
        super(SpiderBrain, self).__init__()
        
        #a SpiderPhysiology object passed in from the world so the brain can access it's body.
        self.physiology = physiology
        self.allNodes = self.physiology.nodes
        
        #an initialized ExplorersHQ object
        #TODO maybe have this class take care of initializing this eventually,
        #       but the init is messy atm, so just pass it in for now.
        self.explorerHQ = explorerHQ
        
        #----<Data Recording Values>----
        #the max amount of experiential points to keep stored. once this value exceeded, the oldest data will be discarded.
        self.dataSize = 5000
        
        #the number of points generated for each moment. i.e. the distance along the time axis data is generated.
        #   TODO we might want to consider lookBack as a max, over which we cast the sensor over n, but at increasing intervals (log probs) as max is approached
        self.lookBack = 100
        #this is the column that will attached to the new data entries to mark the time axis.
        self.lookBackColumn = np.expand_dims(np.arange(self.lookBack)*-1., 1)
        
        #the interval at which the current time step, cts, is updated.
        self.stepInterval = 0.01
        
        #delta time aggregated, the counter
        self._dtAgg = 0
        
        #In summary: for each call to self.step_data, self.lookBack points are generated, where the dt between each is self.timestep.
        #----</Data Recording Values>----
        
        #define the data holding arrays in the name of pre-allocation.
        #   It defines:
        #       1. self.x_timeSeries
        #       2. self.y_timeSeries
        #       3. self.x_data
        #       4. self.y_data
        #       5. self._controlIndices
        self.__define_series()
        
        
    def __define_series(self):
        """Preps arrays in the name of pre-allocation.
            1. A series of control features and environmental features: x
            2. A series of sensory features: y
            3. A data store holding x data cast over the time axis.
            4. A data store holding y corresponding to rows in #3.
            5. An array of indices corresponding to control features in x
           Also verifies node data shapes and numpyness.
        """
        
        xWidth = 0
        yWidth = 0
        controlIndices = []
        
        #iterate nodes to get the number of array columns needed.
        for node in self.allNodes:
            cts_data = node.get_data()
            
            c = cts_data['control'].size  # control features
            e = cts_data['environmental'].size  # environmental features
            s = cts_data['sensory'].size  # sensory features
            
            #in _update_series, control features are recorded first from each node.
            controlIndices.extend(range(xWidth, c))
            
            xWidth += c + e
            yWidth += s
        
        #in _generate_data_over_time, the lookbackColumn is added last
        #   while not properly a control feature, the explorers make decisions based on the time axis.
        #   anyway, it is a pseudo control feature, not an environmental feature.
        #   see the wiki for more details.
        controlIndices.append(xWidth)
        
        self._controlIndices = np.array(controlIndices)
        self.explorerHQ.update_controlIndices(self._controlIndices)
        
        #Use np.zeros, defaulting to float dtype, so in-place modification can occur.
        #create empty time series arrays using the specified series size, and inferred widths.
        self.x_timeSeries = np.zeros((self.lookBack, xWidth))
        self.y_timeSeries = np.zeros((self.lookBack, yWidth))
        
        #create empty arrays using the specified data size and inferred widths.
        #   x_data will hold control and environmental features (+xWidth)
        #       AND the time axis (+1) marking the number of timeIntervals until the reported sensor value actualized.
        #   y_data will only hold sensory feature values paired with x values and sensor values that are cast over time.
        self.x_data = np.zeros((self.dataSize, xWidth+1))
        self.y_data = np.zeros((self.dataSize, yWidth))
    
    def step(self, dt):
        """Records data, and acts.
            This method, and the ones it calls, (I think/hope) do not re-allocate memory to the data arrays.
        """
        
        #check elapsed time.
        self._dtAgg += dt
        if self._dtAgg < self.stepInterval:
            return
        else:
            self._dtAgg = 0
        
        #record data
        self._update_series(dt)
        self._generate_data_over_time()
        
        #get action
        self._act()
    
    def _shift(self, array, shift=1):
        #FIXME 38bgkdlsien this allocates an entire copy. can't want this. need a way to roll without reallocating.
        return np.roll(array, shift=shift, axis=0)
        
    def _update_series(self, dt):
        """Updates both series with current time step data.
        """
        #---<Prep Series>---
        self.x_timeSeries = self._shift(self.x_timeSeries)
        self.y_timeSeries = self._shift(self.y_timeSeries)
        #---</Prep Series>---
        
        #---<Update Series[0]>---
        x_cLoc = 0 #track the column location for slicing.
        y_cLoc = 0
        for node in self.allNodes:
            #step nodes and retrieve the data.
            cts_data = node.step_and_get_data(dt)
            
            #get cts_data sizes
            c = cts_data['control'].size  # control features
            e = cts_data['environmental'].size  # environmental features
            s = cts_data['sensory'].size  # sensory features
            
            #set control feature values
            self.x_timeSeries[0,x_cLoc:x_cLoc+c] = cts_data['control']
            x_cLoc += c
            
            #set environmental feature values
            self.x_timeSeries[0,x_cLoc:x_cLoc+e] = cts_data['environmental']
            x_cLoc += e
            
            #set sensory feature values
            self.y_timeSeries[0,y_cLoc:y_cLoc+s] = cts_data['sensory']
            y_cLoc += s
        #---</Update Series[0]>---
    
    def _generate_data_over_time(self):
        """Updates x_data and y_data, pairing points over the time axis.
            
            After discarding old data, sets x data to both series,
                and sets y data to the current y value for all the new x points,
                effectively casting sensor data over control and environmental data.
        """
        #---<Prep data store>---
        #shift out old data, and turn the first lookBack rows into 1s
        self.x_data = self._shift(self.x_data, shift=self.lookBack)
        self.y_data = self._shift(self.y_data, shift=self.lookBack)
        #---</Prep data store>---
        
        #collect widths
        xWidth = self.x_timeSeries.shape[1]
        yWidth = self.y_timeSeries.shape[1]
        
        #assign the timeSeries array to the first lookBack rows of data, and to the xWidth column.
        self.x_data[0:self.lookBack,0:xWidth] = self.x_timeSeries  # assign x vals
        self.x_data[0:self.lookBack,xWidth:xWidth+1] = self.lookBackColumn # assign time axis vals
        
        #assign the first lookBack rows of y_data to the most recent value in the y series.
        #   this requires adding a dimension to broadcast over the 1s
        self.y_data[0:self.lookBack] = np.expand_dims(self.y_timeSeries[0], 0)
    
    def _act(self):
        """Integrates the explorersHQ
        """
        #FIXME TODO FIXME TODO this method is currently in a pre-alpha state, and is currently implemented for testing purposes.
        
        #TODO update explorerHQ forward model params if new training received. for now, just have the passed instance be fully trained.
        
        #TODO have this come in from the UI, or from more abstract explorerHQs
        #       this will need to ultimately come from outside, UI or genetics (physiology).
        #       and trickle down to dependent explorerHQs
        #update the sensor goal
        self.explorerHQ.update_sensorGoal(np.array([-0.8]))
        
        #update environ info
        # retrieve the most current situation, but add a filler dimension on the end for the time axis.
        x = np.hstack((self.x_timeSeries[0], np.array([0])))
        
        self.explorerHQ.update_environs_only(x)
        
        #step explorers toward goals.
        self.explorerHQ.step_explorer()
        
        #TODO get the explorer control feature location, and set the node control features to those values.
        
    def data_to_npz(self, dest):
        """Writes an npz file holding data
        """
        if os.path.isfile(dest):
            existing = np.load(dest)
            try:
                if (existing['x'].shape[1:] == self.x_data.shape[1:]) and (existing['y'].shape[1:] == self.y_data.shape[1:]):
                    plus_x = np.vstack((existing['x'], self.x_data))
                    plus_y = np.vstack((existing['y'], self.y_data))
                    np.savez(dest, x=plus_x, y=plus_y)
                else:
                    raise EnvironmentError(str(dest) + " contains incompatible spider data for appending. Move or delete it.")
            except KeyError:
                raise EnvironmentError(str(dest) + " is not a spider data file. Move or delete it.")
        else:
            np.savez(dest, x=self.x_data, y=self.y_data)
    