import math
from typing import Tuple

def adjust_frame_dimensions(frame_width: int, frame_height: int) -> Tuple[int, int]:
    """
    Adjusts the frame dimensions to be multiples of 32, rounding up.

    Args:
        frame_width: The original width of the frame.
        frame_height: The original height of the frame.

    Returns:
        A tuple (target_width, target_height) where both dimensions are
        the smallest multiples of 32 that are greater than or equal to
        the original dimensions.
    """
    target_width = math.ceil(frame_width / 32) * 32
    target_height = math.ceil(frame_height / 32) * 32
    
    # Ensure the dimensions are integers
    target_width = int(target_width)
    target_height = int(target_height)

    return target_width, target_height

if __name__ == '__main__':
    # Example Usage and Tests
    test_cases = [
        ((1920, 1080), (1920, 1088)),
        ((1280, 720), (1280, 736)),
        ((640, 480), (640, 480)),
        ((100, 200), (128, 224)),
        ((32, 32), (32, 32)),
        ((33, 33), (64, 64)),
        ((0, 0), (0, 0)), # Edge case: zero dimensions
        ((1, 1), (32, 32)),   # Edge case: minimal dimensions
    ]

    for i, (inputs, expected_output) in enumerate(test_cases):
        frame_w, frame_h = inputs
        result = adjust_frame_dimensions(frame_w, frame_h)
        print(f"Test Case {i+1}: Input ({frame_w}, {frame_h})")
        print(f"Expected: {expected_output}, Got: {result}")
        assert result == expected_output, f"Test Case {i+1} Failed: Expected {expected_output}, but got {result}"
        print("-" * 30)

    print("All test cases passed!")