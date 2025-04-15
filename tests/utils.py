import cv2
import numpy as np

def isolate_red_shapes_from_rgb(image_rgb: np.ndarray, background_color=(255, 255, 255)) -> np.ndarray:
    """
    Keeps only red shapes in an RGB image and replaces everything else with a solid background color.

    Args:
        image_rgb (np.ndarray): Input image in RGB format.
        background_color (tuple): RGB tuple for background (default: white).

    Returns:
        np.ndarray: Image with only red parts retained, rest filled with background color.
    """
    if image_rgb is None or not isinstance(image_rgb, np.ndarray):
        raise ValueError("Input must be a valid NumPy image array in RGB format.")

    # Convert RGB to BGR (OpenCV uses BGR internally)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Convert to HSV
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Define red in HSV (both low and high hue ranges)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Combine both red masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Prepare background in RGB
    result_rgb = np.full_like(image_rgb, background_color, dtype=np.uint8)
    result_rgb[red_mask > 0] = image_rgb[red_mask > 0]

    return result_rgb
