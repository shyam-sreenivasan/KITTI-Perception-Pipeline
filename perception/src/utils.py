import numpy as np

def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to native Python types.
    
    Args:
        obj: Object to convert (dict, list, numpy type, etc.)
        
    Returns:
        Converted object with native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj