import numpy as np
import cv2
import matplotlib.pyplot as plt 
from skimage.metrics import structural_similarity as ssim
import glob

def compute_iou(image1, image2, threshold=128):
    """
    Computes the Intersection over Union (IoU) between two images.

    Args:
        image1 (ndarray): First grayscale or binary image.
        image2 (ndarray): Second grayscale or binary image.
        threshold (int): Threshold to binarize the images.

    Returns:
        float: IoU score.
    """
    assert image1.shape == image2.shape, f"Shape mismatch: {image1.shape} vs {image2.shape}"

    # Binarize images explicitly
    bin1 = image1 > threshold
    bin2 = image2 > threshold

    intersection = np.logical_and(bin1, bin2).sum()
    union = np.logical_or(bin1, bin2).sum()

    print(f"Intersection: {intersection}, Union: {union}")
    return intersection / union if union != 0 else 0


def compute_mse(image1, image2):
    """
    Computes the Mean Squared Error (MSE) between two images.
    """
    assert image1.shape == image2.shape, f"Shape mismatch: {image1.shape} vs {image2.shape}"
    return np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)

def compute_ssim(image1, image2):
    """
    Computes the Structural Similarity Index (SSIM) between two images.
    """
    assert image1.shape == image2.shape, f"Shape mismatch: {image1.shape} vs {image2.shape}"
    if image1.ndim == 3 and image1.shape[2] == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = image1
        gray2 = image2
    score, _ = ssim(gray1, gray2, full=True)
    return score

def compute_feature_match_score(image1, image2):
    """
    Computes a feature-based similarity score using ORB keypoints.
    Returns the number of good matches as a rough similarity score.
    """
    # Convert to grayscale if needed
    if image1.ndim == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if image2.ndim == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    if des1 is None or des2 is None:
        return 0  # No features to match

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    return len(matches)


if __name__ == "__main__":
    print("Getting images")

    # Load images
    # Find all PNG files starting with 'output'
    output_files = sorted(glob.glob("output_*.png"))

    # Load them as grayscale images
    outputs = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in output_files]
    groundtruth = cv2.imread("groundtruth.png", cv2.IMREAD_GRAYSCALE)

    for i, output in enumerate(outputs):
        plt.imshow(output, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Output {i}")
        plt.show()

    metrics = {
        "IoU": compute_iou,
        "Mean Sqaured Error": compute_mse,
        "Structural Similarity Index": compute_ssim,
        "Feature Match Score": compute_feature_match_score,
    }

    # Compare each output to the ground truth
    for i, output in enumerate(outputs):
        print(f"\nMetrics for output_{i} vs groundtruth:")
        for metric_name, metric_func in metrics.items():
            print(f"{metric_name}: {metric_func(output, groundtruth)}")
        
    print(f"\nMetrics for groundtruh vs groundtruth:")
    for metric_name, metric_func in metrics.items():
        print(f"{metric_name}: {metric_func(groundtruth, groundtruth)}")
    