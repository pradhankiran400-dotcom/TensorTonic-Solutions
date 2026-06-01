import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    """Compute forget gate: f_t = sigmoid(W_f @ [h, x] + b_f)"""
    input = np.hstack((h_prev,x_t))
    first = np.dot(input,W_f.T)
    f_t = sigmoid(first + b_f)
    
    return f_t