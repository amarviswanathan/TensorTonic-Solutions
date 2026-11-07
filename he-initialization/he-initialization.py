import math


def he_initialization(W: list, fan_in: int) -> list:
    """
    Returns the weights mapped to the He uniform range.
    """
    L = math.sqrt(6/fan_in)
    return [[w * 2 * L - L for w in row] for row in W]