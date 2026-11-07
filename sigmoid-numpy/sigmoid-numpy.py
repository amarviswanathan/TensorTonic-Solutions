import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # Stable Sigmoid
    x = np.asarray(x,dtype=np.float64)
    out = np.zeros_like(x,dtype=np.float64)

    pos_mask = (x >= 0)
    neg_mask = (x < 0)

    out[pos_mask] = 1/(1 + np.exp(-x[pos_mask]))

    # rewritten formula for negative
    out[neg_mask] = np.exp(x[neg_mask])/(1 + np.exp(x[neg_mask]))
    
    return out