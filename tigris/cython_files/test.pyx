from __future__ import print_function
import numpy as np
cdef double double_tol = 2e-16 

def find_state_with_tag(tag, tag_list):
    ''' finds the location of state in tag_list using binary search.
    tag_list must be sorted.
    Returns idx such that tag_list[idx] = tag '''
    cdef int bm = 0
    cdef int bM = np.shape(tag_list)[0]-1
    cdef int b = 0
    while(bm <= bM):
        b = int((bM + bm)/2)
        if tag < (tag_list[b] - double_tol):
            bM = b-1
        elif tag > (tag_list[b] + double_tol):
            bm = b+1
        else:
            return b
    print('State with tag %f not found! Tolerance: %f'%(tag, double_tol))
    raise NameError('Couldnt find state with tag %f'%tag)
    return None