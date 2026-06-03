import numpy as np

def gini_node(y):
    n = len(y)
    if n <= 1:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return 1.0 - np.sum(probs**2)

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    n_left = len(y_left)
    n_right = len(y_right)
    n_t = n_left + n_right

    if n_t == 0:
        return 0.0

    gini_split = (n_left/n_t)*(gini_node(y_left)) + (n_right/n_t)*(gini_node(y_right))

    return gini_split