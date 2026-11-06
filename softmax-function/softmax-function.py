import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    x = np.array(x)
    shift_x = x - np.max(x,axis=-1,keepdims=True)
    exps = np.exp(shift_x)
    return exps/np.sum(exps,axis=-1,keepdims=True)
    