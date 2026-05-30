import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mean = np.mean(y_true)

    nuem = np.sum((y_true - y_pred)**2)
    den = np.sum((y_true - mean)**2)

    if den == 0:
        return 1.0 if nuem == 0 else 0.0

    answer = 1 - (nuem / den)

    return answer