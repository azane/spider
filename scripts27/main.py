"""this is the main pyglet/pymunk relation file,
        the world/simulation specific stuff will be handled by the various classes of the worlds.py file referenced in pymunk_space
        
        the idea here is that this file should never really have to be changed to render a different world, except to call a different world on one line.
            but maybe even that should be made passable from the command line.
        
        """

import pyglet
from pyglet.window import key, mouse

import pymunk
import spi_pyglet_util

from worlds import *
#FIXME rename "space" in this class to world, cz that's what we call it elsewhere.



class Main(pyglet.window.Window):
    def __init__(self):
        
        pyglet.window.Window.__init__(self, vsync=False)
        
        pyglet.clock.schedule_interval(self.update, 1/60.0)
        
        self.fps_display = pyglet.clock.ClockDisplay()
        
        self.spi_keys = key.KeyStateHandler() #dictionary for storing key states, updated in draw function (as of 20151012, might move), and referred to by world.
        
        
        #Last thing to do.
        self.space = self.spi_pymunk_space()
        #should be nothing below this in this __init__ function.
    
    def spi_pymunk_space(self):
        
        #use a return/assign model so that only ever one world can be selected as the space.
        #FIXME clunky: send window height/width to world.
        return ConveyorTest(self.spi_keys, self.height, self.width)
        
    def update(self,dt):
        steps = 10
        for x in range(steps):
            self.space.step(1/60./steps)
        
        self.space.spi_timestep(dt)
    
    def spi_update_window_to_space(self):
        
        #update window information for custom space.
        #FIXME clunky: is there a better way to do this? you can't pass this to the world, because this owns the world, right?
        #FIXME at least give the world a function by which these are updated, rather than modifying them directly.
        self.space.spi_wh = self.height
        self.space.spi_ww = self.width
    
    def on_draw(self):
        #overriding the window draw update class.
        self.clear()
        
        self.spi_update_window_to_space()
        
        self.fps_display.draw()
        spi_pyglet_util.draw(*self.space.spi_draw_these())
        
        #get keyboard state. The world will already have a reference to this dict. #FIXME should this go in the draw function?
        self.push_handlers(self.spi_keys)
        self.space.spi_control() #the function that does things to the world with key presses.


if __name__ == '__main__':
    main = Main()
    pyglet.app.run()