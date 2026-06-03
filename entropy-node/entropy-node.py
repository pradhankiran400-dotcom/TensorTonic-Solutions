import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    n = len(y)
    if n <= 1:
        return 0.0

    else:
        new_y = np.unique(y)
        counts = len(y)
        counts_unique = len(new_y)
        h_s = 0.0
        for i in range (0,counts_unique):
            class_count = np.sum(y == new_y[i])
            prob = (class_count/counts)
            h_s = h_s - (prob*np.log2(prob))
           
        return h_s
            