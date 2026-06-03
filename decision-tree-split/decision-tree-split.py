import numpy as np

# --- Helper Functions ---
def gini_node(y):
    """Calculates Gini impurity for a single node."""
    n = len(y)
    if n <= 1:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / n
    return 1.0 - np.sum(probs**2)

def gini_impurity(y_left, y_right):
    """Calculates weighted Gini impurity for a split."""
    n_left = len(y_left)
    n_right = len(y_right)
    n_t = n_left + n_right
    
    if n_t == 0:
        return 0.0
        
    return (n_left/n_t) * gini_node(y_left) + (n_right/n_t) * gini_node(y_right)

# --- Main Split Function ---
def decision_tree_split(X, y):
    """
    Finds the best feature and threshold to split the data.
    Returns: [feature_index, threshold]
    """
    X = np.array(X)
    y = np.array(y)
    
    n_samples, n_features = X.shape
    
    best_feature = -1
    best_threshold = None
    min_gini = float('inf')
    
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        
        unique_vals = np.sort(np.unique(feature_values))
        
        
        for i in range(1, len(unique_vals)):
            threshold = (unique_vals[i-1] + unique_vals[i]) / 2.0
            
        
            left_mask = feature_values <= threshold
            right_mask = feature_values > threshold
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            
            current_gini = gini_impurity(y_left, y_right)
            
          
            if current_gini < min_gini:
                min_gini = current_gini
                best_feature = feature_idx
                best_threshold = float(threshold)
                
    return [best_feature, best_threshold]