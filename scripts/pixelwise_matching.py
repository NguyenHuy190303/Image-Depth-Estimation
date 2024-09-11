import cv2
import numpy as np

def l1_distance(x, y):
    return abs(x - y)

def l2_distance(x, y):
    return (x - y) ** 2

def pixel_wise_matching(left_img_path, right_img_path, 
                        similiarity_type,
                        disparity_range,
                        scale=16):
    # Read left, right images then convert to grayscale
    left  = cv2.imread(left_img_path, 0)
    right = cv2.imread(right_img_path, 0)

    left  = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    # Create blank disparity map
    depth = np.zeros((height, width), np.uint8)
    max_value = 255

    if similiarity_type == 'l1':
        similarity_func = l1_distance
    elif similiarity_type == 'l2':
        similarity_func = l2_distance
    else:
        raise Exception('Similarity function not found!')

    # Precompute the cost for all disparities
    costs = np.full((height, width, disparity_range), max_value, dtype=np.float32)
    for j in range(disparity_range):
        left_d = left[:,j:width]
        right_d = right[:,0:width-j]
        costs[:, j:width, j] = similarity_func(left_d, right_d)

    # Find the disparity with the minimum cost
    min_cost_indices = np.argmin(costs, axis=2)

    # Set the disparity map
    depth = min_cost_indices * scale
    depth = depth.astype(np.uint8)

    return depth, cv2.applyColorMap(depth, cv2.COLORMAP_JET)