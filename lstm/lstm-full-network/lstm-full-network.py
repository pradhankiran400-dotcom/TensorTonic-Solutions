import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_c = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """
        Forward pass. Returns (y, h_last, C_last).
        Assumes X has shape (N, T, D) where:
          N = Batch Size
          T = Sequence Length (Time steps)
          D = Input Dimension
        """
        N, T, D = X.shape
        
        h_t = np.zeros((N, self.hidden_dim))
        c_t = np.zeros((N, self.hidden_dim))
        
        output_dim = self.W_y.shape[0]
        y = np.zeros((N, T, output_dim))
        
        for t in range(T):

            x_t = X[:, t, :]
            

            concat_input = np.hstack((h_t, x_t))
            
            # 2. Gate calculations
            f_t = sigmoid(np.dot(concat_input, self.W_f.T) + self.b_f)
            i_t = sigmoid(np.dot(concat_input, self.W_i.T) + self.b_i)
            C_tilde = np.tanh(np.dot(concat_input, self.W_c.T) + self.b_c)
            o_t = sigmoid(np.dot(concat_input, self.W_o.T) + self.b_o)
            
            # 3. State updates
            c_t = f_t * c_t + i_t * C_tilde
            h_t = o_t * np.tanh(c_t)
            
            y_t = np.dot(h_t, self.W_y.T) + self.b_y
            
            y[:, t, :] = y_t
            
       
        return (y, h_t, c_t)