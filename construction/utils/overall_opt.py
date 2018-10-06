
'''------------------------------  Instruction -------------------------------
When using this code, you need to prepare data in the following shape: demand starts from the ealiest day (indexed as 0);
Orders needs all order create_tm_index
-----------------------------------------------------------------------------'''
#############################################################################################
#                    for End2End Model, compute opt DP order use long term
#############################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




h = 1
b = 9

def get_obj_val(d,i_a):
    # input: a vector d which is the demand from 
    val = 0
    for i in range(len(d)):
        val = val + np.maximum(h*(i_a -np.sum(d[0:i+1])),0) + np.maximum(b*(np.sum(d[0:i+1])-i_a),0)
        # print('val updated to', val)
    return val
    
def opt_order(d, inv):
    pre_val = 10000000
    pre_order = -100
    if len(d) == 0:
#        print('d zero length!')
        return 0
    for i in (np.array(range(len(d)))[::-1]):
        temp_val = get_obj_val(d , np.sum(d[0:i+1]))
        temp_order = np.sum(d[0:i+1]) -inv
        print(i,temp_val,temp_order)
        if (temp_val > pre_val ):
#            print('checked point ',i,'val is', temp_val)
#            print('temp_val', temp_val,'temp_order', temp_order)
#            print('pre_val', pre_val,'pre_order', pre_order)
            return np.maximum(pre_order,0)
        else:
#            print('checked point ',i,'val is', temp_val)
            pre_val = temp_val
            pre_order = temp_order
#    print('keep decreasing')  
    return np.maximum(temp_order,0)


dm = [10,9,8,7,6,5,4,3,2,1,10]
# dm = dm[::-1]
inv = 0
s = opt_order(dm, inv)
print(s)

