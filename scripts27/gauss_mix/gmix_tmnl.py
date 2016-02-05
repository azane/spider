"""Gaussian Mixture model terminal access."""

import numpy as np
import tensorflow as tf
import sys as sys


"""
command line calling structure
:python thisfile.py <str:train.npz> <str:test.npz>
   where train.npz and test.npz should have two arrays, x and t.
   and x.shape == (s,inDims)
   and t.shape == (s,outDims)
train and test should obviously be different sets of data from the same source.

"""


XMVU_PATH = "/Users/azane/GitRepo/spider/data/xmvu.npz" #file to which the xmvu data, for sampling is written after training.

#----------------------------------------------------------------<MAIN>----------------------------------------------------------------
if __name__ == "__main__":
    #TODO TODO split up this main section. make a function that can be called just with x and t array values
    #       cz the only reason to really use the command line is to read from a file...so that's really the only thing we need in the main section.
    
    #----<Read Data>----
    #t for test, s for training sample
    s_x, s_t = get_xt_from_npz(sys.argv[1])
    t_x, t_t = get_xt_from_npz(sys.argv[2])
    #----</Read Data>----
    
    #----<Training Constants>----
    ITERATIONS = range(1000) #iterable
    
    TEST_BATCH_SIZE = 500
    if t_x.shape[0] < TEST_BATCH_SIZE:
        TEST_BATCH_SIZE = t_x.shape[0]
        
    TRAIN_BATCH_SIZE = 1000
    if s_x.shape[0] < TRAIN_BATCH_SIZE:
        TRAIN_BATCH_SIZE = s_x.shape[0]
    #----</Training Constants>----
    
    #----<Training Setup>----
    inDims, outDims = infer_space(s_x, s_t) #get info to build model
    modelDict = gmix_training_model(inDims, outDims) #build model, get dict of tensorflow placeholders, variables, outputs, etc.
    
    inRange, outRange = infer_ranges(s_x, s_t) #infer ranges for feed dict
    feed_dict = {
                    modelDict['x']:None,
                    modelDict['t']:None,
                    modelDict['inRange']:inRange,
                    modelDict['outRange']:outRange
                } #build feed dict, but leave x and t empty, as they will be updated in the training loops.
    #----</Training Setup>----
    
    #----<Training Loop>----
    for i in ITERATIONS:
        
        if i % 10 == 0: #run reports every 10 iterations.
            feed_dict[modelDict['x']], feed_dict[modelDict['t']] = sample_batch(t_x, t_t, TEST_BATCH_SIZE) #update feed_dict with test batch
            
            result = modelDict['sess'].run([
                                            modelDict['summaries'],
                                            modelDict['loss']
                                        ], feed_dict=feed_dict) #run model with test batch
            
            modelDict['summaryWriter'].add_summary(result[0], i) #write to summary
            print("Loss at step %s: %s" % (i, result[1])) #print loss
            
            print '-------------------------------'
            
        else:
            #update feed_dict with training batch
            feed_dict[modelDict['x']], feed_dict[modelDict['t']] = sample_batch(s_x, s_t, TRAIN_BATCH_SIZE)
            modelDict['sess'].run([
                                    modelDict['train_step']
                                ], feed_dict=feed_dict) #run train_step with training batch
    #----<Training Loop>----
    
    #----<Store for Sampling>----
    feed_dict[modelDict['x']], feed_dict[modelDict['t']] = t_x, t_t #update feed_dict with full test data batch
    #evaluate the trained model at m, v, and u with the test data.
    result = modelDict['sess'].run([
                                    modelDict['m'],
                                    modelDict['v'],
                                    modelDict['u']
                                ],feed_dict=feed_dict)
    np.savez(XMVU_PATH, x=t_x, m=result[0], v=result[1], u=result[2]) #write this to an npz file for later sampling
    #----</Store for Sampling>----
