        

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

    Example:
        >>> greet("Alice", 3)
    
    Caution:
        Ensure that the 'times' parameter is not negative.

    notes:
        This function is a simple demonstration of how to use docstrings.
    
    See Also:
        - `calculate_area`: A function that calculates the area of a circle.
    
    Warning:
        This function does not handle non-string names gracefully.

    \Ref: greet
    """
    if times < 0:
        raise ValueError("Times cannot be negative")
    for _ in range(times):
        print(f"Hello, {name}!")

# TO check the automatic documentation 
class test:
    """
    A simple test class to demonstrate docstring usage.
    
    Attributes:
        name (str): The name of the test.
        value (int): An integer value associated with the test.
        
    Methods:
        run(): Executes the test and prints a message.
    """
    
    def __init__(self, name, value):
        """
        Initialize the test with a name and value.
        
        Args:
            name (str): The name of the test.
            value (int): An integer value for the test.

        Raises:
            TypeError: If name is not a string or value is not an integer.

        TODO:
            - Add more attributes to the test class.
            - Implement more complex test logic.
        
        FIXME:
            - Handle cases where name is empty.
        
        Hack:
            - Use a default value for value if not provided.
        XXX:
            - Consider renaming the class to something more descriptive.
        """
        self.name = name
        self.value = value

    def run(self):
        """Run the test and print a message."""
        print(f"Running test: {self.name} with value {self.value}")