import cv2
import numpy as np
from collections import defaultdict
import math

def cluster_lines(lines):
    """
    Given a list of lines (start point, end point) seperate it into two list, 
    differentiating based on angle
    """

    # Define criteria = (type, max_iter, epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    attempts = 10
    k = 2

    # returns angles in [0, pi] in radians
    angles = np.array([theta for rho, theta in lines])

    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def intersec_between_groups(lines):
    """
    Compute intersection between two groups of lines
    :param lines: 2D list representing two groups of lines
    :returns : list of all the intersections
    """
    intersections = []
    for l1 in lines[0]:
        for l2 in lines[1]:
            intersections.append(intersection(l1, l2))

    return intersections

def intersection(line1, line2):
    """
    Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    
    try:
        x0, y0 = np.linalg.solve(A, b)
    except Exception: # Singular matrix error
        x0, y0 = np.linalg.lstsq(A, b)[0]
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]

def polar_to_carthesian(rho, theta, distance_factor):
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + distance_factor*(-b)), int(y0 + distance_factor*(a)))
    pt2 = (int(x0 - distance_factor*(-b)), int(y0 - distance_factor*(a)))
    return [pt1, pt2]
