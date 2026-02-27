import numpy as np

def is_contain_nan(x):
    x = np.array(x)
    x = x.flatten()
    for e in x:
        if np.isnan(e):
            return True
    return False