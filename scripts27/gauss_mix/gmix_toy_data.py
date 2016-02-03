import sys
import numpy as np

"""command line: python this.py <str: outfilepath.npz> <int: sample size> <float: variance>"""

#Generate a 2d pile of sample data as seen on page 273 in Bishop's ML book.

def gen_sample(s, v):
    #sample uniform
    x = np.random.uniform(low=0.0, high=1.0, size=s)
    #formulate
    t_ = x + 0.3*np.sin(2*np.pi*x)
    #apply noise
    t = t_ + np.random.normal(0, v, size=t_.shape)
    
    #add dim for consistency with multidemsional data
    x = np.expand_dims(x, 1)
    t = np.expand_dims(t, 1)
    
    #return 2d
    return t, x #reverse t and x. t will be read as the independent data
    #i.e. this should be called like this: x, y = thisfile.gen_sample(size, variance)


if __name__ == "__main__":
    x, y = gen_sample(int(sys.argv[2]), float(sys.argv[3])) #take the second argument as the sample size

    #write file to passed path, first argument.
    np.savez(sys.argv[1], x=x, y=y)