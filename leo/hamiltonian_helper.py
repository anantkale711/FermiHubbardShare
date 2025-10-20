import leo.hamiltonians3 as hamiltonians3
import numpy as np


def create_H_func_new(Hk_u, Hk_d, Hi):
    dimH_up, dimH_down = np.shape(Hi)
    dimH = dimH_up*dimH_down
    global _temp_mat
    _temp_mat = np.zeros((dimH_up, dimH_down))
    def apply_H_on_state_out(state, out = None):
        state_matrix = state.reshape(dimH_up, dimH_down)
        if out is None:
            result = np.zeros(np.shape(state_matrix))
        else:
            out *= 0
            out.shape = (dimH_up, dimH_down)
            result = out
        
        result += Hk_u@state_matrix
        result += state_matrix @ Hk_d.transpose()
        np.multiply(Hi, state_matrix, out = _temp_mat)
        result += _temp_mat

        result.shape = dimH
        return result
    
    return apply_H_on_state_out

def create_H_func_new_mat(Hk_u, Hk_d, Hi):
    dimH_up, dimH_down = np.shape(Hi)
    dimH = dimH_up*dimH_down
    global _temp_mat
    _temp_mat = np.zeros((dimH_up, dimH_down))
    def apply_H_on_state_out(state_matrix, out = None):
        if out is None:
            result = np.zeros(np.shape(state_matrix))
        else:
            out *= 0
            out.shape = (dimH_up, dimH_down)
            result = out
        
        result += Hk_u@state_matrix
        result += state_matrix @ Hk_d.transpose()
        np.multiply(Hi, state_matrix, out = _temp_mat)
        result += _temp_mat

        return result
    
    return apply_H_on_state_out

def apply_operator_on_state_helper(Op, sigma, state_matrix, out_matrix=None):
    ''' Op is the operator to be applied to the state_matrix along axis given by sigma
    sigma: 0 -> up spin (axis 0)
    sigma: 1 -> down spin (axis 1)
    sigma: 2 -> diagonal in fock basis'''
    if sigma == 0:
        result = Op @ state_matrix
    elif sigma == 1:
        result = state_matrix @ Op.transpose()
    else:
        result = np.multiply(Op, state_matrix)
    return result

def apply_operator_list_on_state(Op_list, state_matrix):
    ''' Op_list is an ordered list of tuples (Op,simga) with
    operators Op and sigma = {0,1,2} to be applied in sequence '''
    
    temp_state = state_matrix
    for (Op, sigma) in Op_list[::-1]:  # go through in reverse to apply operators in correct order
        result = apply_operator_on_state_helper(Op, sigma, temp_state)
        temp_state = result
    return result