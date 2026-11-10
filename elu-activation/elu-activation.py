import math
import numpy as np


def elu(x: list, alpha: float = 1.0) -> list:
    """
    Returns ELU applied elementwise to the input values.
    """
    for i,val in enumerate(x):
        if val <= 0:
            x[i] = alpha * (math.exp(val) - 1)
    return x
    