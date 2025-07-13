# TO check the automatic documentation 
def calculate_area(radius):
    """
    Calculate the area of a circle.
    
    Args:
        radius (float): The radius of the circle in meters.
        
    Returns:
        float: The area of the circle in square meters.
        
    Example:
        >>> calculate_area(2.0)
        12.566370614359172
    """
    import math
    return math.pi * radius ** 2


def greet(name, times=1):
    """
    Print a greeting message multiple times.
    
    Parameters:
        name (str): The name to greet.
        times (int): Number of times to repeat the greeting (default: 1).
        
    Raises:
        ValueError: If times is negative.
    """
    if times < 0:
        raise ValueError("Times cannot be negative")
    for _ in range(times):
        print(f"Hello, {name}!")