"""Time based visualizer of 4 dimensions.
The basic class was retrieved from this stackoverflow question: http://stackoverflow.com/questions/9401658/matplotlib-animating-a-scatter-plot
    azane modified it a little, though I regrettably failed to document those changes. : / OOPS!

#FIXME after a brief look at this class, the below may be slightly false...but i'm not a fan of this visualizer, so azane probs won't fix it. : )
built to take csv's generated from spider
    so the last column should be the sensor
    the second to last column should be time
    beyond that, it takes only the first two columns.
        which means that columns after the first two, and before the last two, will be omitted.
"""


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import juggle_axes

FLOOR = 50
CEILING = -50

class AnimatedScatter(object):
    def __init__(self, csv):
        
        #gather data
        self.points = np.genfromtxt(csv, delimiter=',')
        
        #get furthest lookback along n.
        self.t = np.amin(self.points[:,-2]) - 1 #-1 for count, not index. - cz its negative.
        self.t = np.absolute(self.t) #get positive for looping
        self.t = int(self.t) #conver from numpy float to int.
        
        self.stream = self.data_stream() #get iterator from data_stream
        self.angle = 50
        
        self.fig = plt.figure() #init figure
        self.ax = self.fig.add_subplot(111,projection = '3d') #get subplot
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=50, 
                                           init_func=self.setup_plot, blit=True) #set animation on fig
        
        
    def change_angle(self):
        self.angle = (self.angle + 0.2)%360
        
    def setup_plot(self):
        data = next(self.stream)
        
        self.scat = self.ax.scatter(data[0,:], data[1,:], data[2,:], animated=True)
        
        #self.ax.set_xlim3d(FLOOR, CEILING)
        #self.ax.set_ylim3d(FLOOR, CEILING)
        #self.ax.set_zlim3d(FLOOR, CEILING)
        
        return self.scat,
        
    def data_stream(self):
        
        for i in range(self.t): #iterate over lookback.
            #for all rows in self.data where n=-i
            selForN = self.points[np.where(self.points[:,-2] == (-1*(i)))]
            
            selForN = selForN.transpose()
            
            xyz = selForN[[0,1,-1], :]
            
            yield xyz
            
    def update(self, i):
        data = next(self.stream)
        
        self.scat._offsets3d = data
        
        self.change_angle()
        self.ax.view_init(45,self.angle)
        
        plt.draw()
        #plt.show()
        return self.scat,
        
    def show(self):
        plt.show()

if __name__ == '__main__':
    a = AnimatedScatter('data/full_with_n.csv')
    a.show()