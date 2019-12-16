import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import itertools
from scipy.signal import argrelextrema

def detect_corners(edges, threshold, lines_limit=12):
    """
    Given a single-channel image representing the edges, compute HoughLines and find the intersection
    between lines of differente angles. Try all possible combinations of quadrilaters given
    the intersections and find the best one which approximate the edges. Return the four corners of 
    the best polygon found.

    :param edges: Single-channel image
    :param threshold: Threshold used for Hough lines
    :param lines_limit: If more than lines_limit lines are found raise an error. Used to avoid extremely heavy computation due to 4 vertices polygon combinations.
    :returns: A tuple containing 2 values: numpy array of corners and bi-dimensional list of segmented lines
    """
    lines = cv2.HoughLines(edges, 1, np.pi/180.0, threshold)
    if lines is None:
        return None
    lines = lines.squeeze(axis=1)

    # Avoid too much intersections heavy computation
    if lines.shape[0] > lines_limit:
        raise ValueError("Too much lines. Limit is %d" % lines_limit)

    # Differentiate lines in two clusters (horizontal and vertical)
    lines_groups = cluster_lines(lines)
    intersec = np.array(intersec_between_groups(lines_groups), dtype=np.int32).squeeze(axis=1)

    # Differentiate corners in four clusters (top-left, top-right, bottom-right, bottom-left)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(np.array(intersec, dtype=np.float32), 4, None, criteria, 10, 
            cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape(-1)
    cluster = (intersec[labels == 0], intersec[labels == 1], intersec[labels == 2], intersec[labels == 3])

    # For each combination of 4 vertices, compute scores based on edges coverage
    scores = []
    comb = list(itertools.product(*cluster))
    for v in comb:
        polys = np.zeros(edges.shape, dtype=np.int32)
        cv2.polylines(polys, [np.array(v)], True, 255)
        mask = polys[edges > 0] # Assume as input edges any values greater than 0
        scores.append(np.sum(mask))

    # Use as best corners the vertices of the polygon with the best score
    best = np.argmax(scores)
    return (np.array(comb[best]), lines_groups)

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
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

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
