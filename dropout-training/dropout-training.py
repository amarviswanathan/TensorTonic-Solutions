import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    x  = np.asarray(x,dtype=float)

    if not 0 <= p < 1 :
        raise ValueError('p must satisfy 0 <= p < 1')

    keep_prob = 1 - p

    if rng is not None:
        random_values = rng.random(x.shape)
    else:
        random_values = np.random.random(x.shape)


    keep_mask = random_values < keep_prob

    dropout_pattern = keep_mask.astype(float)/keep_prob 

    output = x * dropout_pattern

    
    return output, dropout_pattern