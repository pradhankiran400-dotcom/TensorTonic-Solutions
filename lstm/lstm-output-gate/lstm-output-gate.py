import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def output_gate(h_prev: np.ndarray, x_t: np.ndarray, C_t: np.ndarray,
                W_o: np.ndarray, b_o: np.ndarray) -> tuple:
    """Compute output gate and hidden state."""
    
    # 1. Concatenate inputs along the feature dimension -> shape: (N, H+D)
    input_concat = np.hstack((h_prev, x_t))
    
    # 2. Linear transformation with transposed weights -> shape: (N, H)
    # Using input_concat @ W_o.T is the standard way for batched inputs
    o_t = sigmoid(np.dot(input_concat, W_o.T) + b_o)
    
    # 3. Calculate new hidden state (element-wise multiplication) -> shape: (N, H)
    h_t = o_t * np.tanh(C_t)
    
    return (o_t, h_t)