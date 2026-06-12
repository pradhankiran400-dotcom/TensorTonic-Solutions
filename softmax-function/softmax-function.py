import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.array(x)
    n_d = x.ndim
    
    if (n_d == 1):
        max_val = np.max(x)
        exp_x = np.exp(x - max_val)
        ans = (exp_x/np.sum(exp_x))

        return ans

    elif n_d == 2:

        max_val = np.max(x,axis=1,keepdims=True)

        exp_x = np.exp(x - max_val)

        ans = (exp_x/np.sum(exp_x,axis=1,keepdims=True))

        return ans
            