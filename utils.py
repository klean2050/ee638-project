"""adapted from https://github.com/usc-sail/fed-multimodal/blob/main/features/simulation_features/simulation_manager.py"""

import cv2, numpy as np


def label_noise_matrix(seed, noise, sparsity, class_num=5):
    np.random.seed(seed)
    noisy_level = noise
    sparse_level = sparsity

    prob_matrix = [1 - noisy_level] * class_num * class_num
    sparse_elements = np.random.choice(
        class_num * class_num, round(class_num * (class_num - 1) * sparse_level)
    )
    for idx in range(len(sparse_elements)):
        while sparse_elements[idx] % (class_num + 1) == 0:
            sparse_elements[idx] = np.random.choice(class_num * class_num, 1)
        prob_matrix[sparse_elements[idx]] = 0

    available_spots = np.argwhere(np.array(prob_matrix) == 1 - noisy_level)
    for idx in range(class_num):
        available_spots = np.delete(
            available_spots, np.argwhere(available_spots == idx * (class_num + 1))
        )

    for idx in range(class_num):
        row = prob_matrix[idx * 4 : (idx * 4) + 4]
        if len(np.where(np.array(row) == 1 - noisy_level)[0]) == 2:
            unsafe_points = np.where(np.array(row) == 1 - noisy_level)[0]
            unsafe_points = np.delete(
                unsafe_points,
                np.where(np.array(unsafe_points) == idx * (class_num + 1))[0],
            )
            available_spots = np.delete(
                available_spots, np.argwhere(available_spots == unsafe_points[0])
            )
        if np.sum(row) == 1 - noisy_level:
            zero_spots = np.where(np.array(row) == 0)[0]
            prob_matrix[zero_spots[0] + idx * 4], prob_matrix[available_spots[0]] = (
                prob_matrix[available_spots[0]],
                prob_matrix[zero_spots[0] + idx * 4],
            )
            available_spots = np.delete(available_spots, 0)

    prob_matrix = np.reshape(prob_matrix, (class_num, class_num))
    for idx in range(len(prob_matrix)):
        zeros = np.count_nonzero(prob_matrix[idx] == 0)
        if class_num - zeros - 1 == 0:
            prob_element = 0
        else:
            prob_element = (noisy_level) / (class_num - zeros - 1)
        prob_matrix[idx] = np.where(
            prob_matrix[idx] == 1 - noisy_level, prob_element, prob_matrix[idx]
        )
        prob_matrix[idx][idx] = 1 - noisy_level

    return prob_matrix

def increase_contrast(image):
    # Parameters for manipulating the image
    copy = image.copy()
    maxIntensity = 255.0
    phi, theta = 1.3, 1.5
    # Decrease intensity such that dark pixels become
    # much darker, and bright pixels become slightly dark.
    copy = (maxIntensity/phi)*(copy/(maxIntensity/theta))**2
    return np.array(copy, dtype=int)


def find_contours(image):
    # Increase constrast in image to increase
    # chances of finding contours.
    processed = increase_contrast(image)
    # Get the grayscale of the image.
    processed = np.uint8(processed)
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    # Detect contour(s) in the image.
    cnts = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # At least ensure that some contours were found.
    if len(cnts):
        # Find the largest contour in the mask.
        c = max(cnts, key=cv2.contourArea)
        (_, radius) = cv2.minEnclosingCircle(c)
        # Assume the radius is of a certain size.
        if radius > 100:
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            return (center, radius)
    return None
        
        
def resize_and_center_fundus(image, diameter):

    copy = image.copy()
    contours = find_contours(copy)
    if contours is None:
        return None
    center, radius = contours
    
    # Calculate the min and max-boundaries for cropping the image.
    x_min = max(0, int(center[0] - radius))
    y_min = max(0, int(center[1] - radius))
    z = int(radius*2)
    x_max = x_min + z
    y_max = y_min + z

    copy = copy[y_min:y_max, x_min:x_max]
    # Scale the image.
    fx = fy = (diameter / 2) / radius
    copy = cv2.resize(copy, (0, 0), fx=fx, fy=fy)

    # Add padding.
    shape = copy.shape
    top = bottom = int((diameter - shape[0])/2)
    left = right = int((diameter - shape[1])/2)
    if shape[0] + top + bottom == diameter - 1:
        top += 1
    if shape[1] + left + right == diameter - 1:
        left += 1

    # Define border of the image.
    border = [top, bottom, left, right]
    return cv2.copyMakeBorder(
        copy, *border, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )