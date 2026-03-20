import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    #change to np array
    x = np.array(x,dtype=float)
    p = np.array(p,dtype=float)

    #check if x and p match 
    if x.shape != p.shape:
        raise ValueError("shapes of x, and p must match")
        return
    #check if p = 1.0 and within tolerance 10^-6
    if np.allclose(np.sum(p),1.0,atol=1e-6) != 1:
        raise ValueError("probabilities must sum to 1")
        return
    # E[X] = sum of x and p
    return np.sum(x*p)
    pass
