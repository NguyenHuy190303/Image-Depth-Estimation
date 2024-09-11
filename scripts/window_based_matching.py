import cv2
import numpy as np

from scipy.signal import convolve2d

def l1_distance(x, y):
    return -1 * np.sum(abs(x - y))

def l2_distance(x, y):
    return -1 * np.sqrt(np.sum((x - y) ** 2))
    
def cosine_similarity(x, y):
    numerator = np.dot(x, y)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)

    return numerator / denominator if denominator != 0 else 0

def correlation_coefficient(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))
    
    return numerator / denominator if denominator != 0 else 0

def window_based_matching(left_img_path, right_img_path, 
                          similiarity_type,
                          disparity_range, 
                          kernel_size=5, 
                          scale=16):
    left  = cv2.imread(left_img_path, 0)
    right = cv2.imread(right_img_path, 0)

    left  = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    # Create blank disparity map
    depth = np.zeros((height, width), np.uint8)
    kernel_half = int((kernel_size - 1) / 2)

    if similiarity_type == 'l1':
        similarity_func = l1_distance
    elif similiarity_type == 'l2':
        similarity_func = l2_distance
    elif similiarity_type == 'cosine':
        similarity_func = cosine_similarity
    elif similiarity_type == 'correlation':
        similarity_func = correlation_coefficient

    for y in range(kernel_half, height-kernel_half):
        for x in range(kernel_half, width-kernel_half):
            # Find j where cost has minimum value
            disparity = 0
            cost_optimal = -10000

            for j in range(disparity_range):
                d = x - j
                cost = -10000
                if (d - kernel_half) > 0:
                    wp = left[(y-kernel_half):(y+kernel_half)+1, (x-kernel_half):(x+kernel_half)+1]
                    wqd = right[(y-kernel_half):(y+kernel_half)+1, (d-kernel_half):(d+kernel_half)+1]

                    wp_flattened = wp.flatten()
                    wqd_flattened = wqd.flatten()

                    cost = similarity_func(wp_flattened, wqd_flattened)

                if cost > cost_optimal:
                    cost_optimal = cost
                    disparity = j

            depth[y, x] = disparity * scale

    depth = depth.astype(np.uint8)

    return depth, cv2.applyColorMap(depth, cv2.COLORMAP_JET)
