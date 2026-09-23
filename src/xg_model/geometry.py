import math

def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return the Euclidean distance between two points."""
    return math.hypot(x2 - x1, y2 - y1)