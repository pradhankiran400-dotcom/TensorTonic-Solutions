import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x, dtype=float)

    if p == 0.0:
        mask = np.ones(x.shape)
        return (x, mask)
    else:
        if rng is not None:
            rand_vals = rng.random(x.shape)
        else:
            rand_vals = np.random.random(x.shape)
            
        binary_mask = (rand_vals >= p).astype(float)
        
        dropout_pattern = binary_mask / (1 - p)
        
        output = x * dropout_pattern
        
        return (output, dropout_pattern)