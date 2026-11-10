import numpy as np

def epsilon_greedy(q_values: list, epsilon: float, seed: int = 0) -> int:
    """
    Returns the action index as an integer.
    """
    rng = np.random.default_rng(seed)
    q_values = np.asarray(q_values,dtype=np.float64)
    
    if rng.random() < epsilon:
        return rng.integers(0,len(q_values))
    else:
        return np.argmax(q_values)
    pass