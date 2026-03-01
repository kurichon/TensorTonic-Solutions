import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    np_array = np.asarray(x,dtype=float)
    sigmoid_val = 1 / (1 + np.exp(-np_array) )
    return sigmoid_val
    # Write code here
    pass