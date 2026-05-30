import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.array(x)
    answer = 1/(1+np.exp(-x))
    return answer.tolist()