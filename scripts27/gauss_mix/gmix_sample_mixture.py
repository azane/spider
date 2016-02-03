import sys as sys
import numpy as np
import tensorflow as tf


"""terminal syntax

: python thisfile.py xmvuinfile.npz sampleoutfile.npz

the xmvu file must have 4 arrays labeled x, m, v, and u, and must meet the asserted requirements in the sample_mixture function.
"""

def sample_mixture(x, m, v, u):
    #x - the inputs [s,x] #these are strictly used to pair with the returned outputs.
    #u - the mean [s,g,t]
    #v - the variance [s,t]
    #m - the mixing coefficients [s,g]
    
    #assert that s,g and t/x match everywhere.
    s = u.shape[0]
    g = u.shape[1]
    t = u.shape[2]
    assert s == x.shape[0] and s == v.shape[0] and s == m.shape[0]
    assert g == m.shape[1]
    assert t == v.shape[1]
    #TODO ? assert that all mixtures sum to 1?
    
    #expand dims for broadcasting
    v = np.expand_dims(v, 1) #v.shape == (s,1,t)
    
    #sample all distributions for all targets, given corresponding v,u
    vec_normal = np.vectorize(np.random.normal) #vectorize for broadcasted operations.
    premix = vec_normal(u, v) #premix.shape == (s,g,t)
    print 'premix.shape: ' + str(premix.shape)
    
    #create output array y, this will be overwritten with target values.
    y = np.zeros((s,t))
    
    #use the mixing coefficients (0-1) as the likelihood that a given gaussian's samples will be selected for a given input.
    for i in range(s):
        gSelection = np.random.choice(g, p=m[i]) #pick over range g with the mixtures for this sample.
        #for this sample, select the target array of this gSelection
        y[i] = premix[i,gSelection]
    
    print 'y.shape: ' + str(y.shape)
    
    return x, y

if __name__ == "__main__":
    
    #when calling from the command line, pass the path to the .npy file containing four arrays labeled x, m, v, and u.
    with np.load(sys.argv[1], allow_pickle=False) as xmvu:
        x = xmvu['x']
        m = xmvu['m']
        v = xmvu['v']
        u = xmvu['u']
    
    #write the returned arrays of shape [s,x] and [s,t] to a .npy file to the path specced by sys.argv[2]
    s_x, s_y = sample_mixture(x, m, v, u)
    
    np.savez(sys.argv[2], x=s_x, y=s_y)