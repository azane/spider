import sys
import numpy as np

#Generate a 2d pile of sample data as seen on page 273 in Bishop's ML book.

def gen_sample(s):
    #sample uniform
    x = np.random.uniform(low=0.0, high=1.0, size=s)
    #formulate
    t_ = x + 0.3*np.sin(2*np.pi*x)
    #apply noise
    t = t_ + np.random.normal(0, 0.05)
    
    #return 2d
    return np.stack((t, x), axis=-1) #reverse t and x. t will be read as input.

s = 5000 #sample size
t = 500 #test size

sample = gen_sample(s)
test = gen_sample(t)

#write files to passed directory
np.savetxt((sys.argv[1] + '/sample.csv'), sample, delimiter=',')
np.savetxt((sys.argv[1] + '/test.csv'), test, delimiter=',')