import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    x = np.array(x,dtype=float)
    ans = []
    for i in x:
        if(i>=0):
            item = i
            ans.append(item)

        else:
            item = alpha*i
            ans.append(item)

    return np.array(ans)