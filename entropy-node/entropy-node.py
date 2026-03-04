import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.array (y,dtype=float)
    entropy = 0.0
    n = len(y)

    unique_values, counts = np.unique(y, return_counts=True)
    
    for count in counts:
        p = count / n
        if p > 0:
            entropy -= p * np.log2(p)

    return entropy