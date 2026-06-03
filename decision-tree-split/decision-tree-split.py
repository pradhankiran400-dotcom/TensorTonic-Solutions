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
    
    # 1. Loop through each feature
    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        
        # Sort unique values to find consecutive midpoints
        unique_vals = np.sort(np.unique(feature_values))
        
        # 2. Try thresholds at the midpoint between consecutive unique values
        for i in range(1, len(unique_vals)):
            threshold = (unique_vals[i-1] + unique_vals[i]) / 2.0
            
            # Split the data based on the threshold
            left_mask = feature_values <= threshold
            right_mask = feature_values > threshold
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            # 3. Weight the child Gini impurities by the fraction of samples
            current_gini = gini_impurity(y_left, y_right)
            
            # Update the best split if this one is better
            if current_gini < min_gini:
                min_gini = current_gini
                best_feature = feature_idx
                best_threshold = float(threshold) # Ensure it's a standard float
                
    # 4. Return as a list
    return [best_feature, best_threshold]