import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import itertools
from scipy.signal import argrelextrema
from abc import ABC, abstractmethod
import time
from koda.edge.network import UNetEdgeDetector

class CornersDetector(ABC):

    @abstractmethod
    def find_corners(self, img, iterations=3):
        """
        #TODO
        """
        pass

class CornersNotFound(RuntimeError):
        def __init__(self, message):
            super().__init__(message)

class CornersDetectorByEdges(CornersDetector):
    def __init__(self, timeout_ns=800*(10**6)):
        super().__init__()
        self.noise_threshold = 80
        self.hough_threshold = 60
        self.hough_resolution = (1, np.pi/36)
        self.start_ns = None
        self.timeout_ns = timeout_ns
        self.timed_out = False
        self.edge_detector = UNetEdgeDetector()
        self.edge_detector.load_model('koda/unet-70.h5')

    def find_corners(self, img, iterations=3):
        self.timed_out = False
        self.start_ns = time.time_ns()

        # Detect edge
        img = self.edge_detector.evaluate(img)

        # Cleanup spourius pixel from edge detection model
        img[img < self.noise_threshold] = 0

        # Get corners, adjust params for each iterations tries specified
        e = None
        for _ in range(iterations + 1):
            try:
                corners, lines = self.detect_corners(img, self.hough_threshold, hough_res=self.hough_resolution)
            except CornersNotFound as e1:
                # Too few line, try to decrease threshold
                self.hough_threshold = int(self.hough_threshold - (self.hough_threshold*0.1))
                e = e1
                continue
            break

        if e is not None:
            raise e

        return corners

    def detect_corners(self, edges, threshold, hough_res=(1, np.pi/180)):
        """
        Given a single-channel image representing the edges, compute HoughLines and find the intersection
        between lines of differente angles. Try all possible combinations of quadrilaters given
        the intersections and find the best one which approximate the edges. Return the four corners of 
        the best polygon found.

        :param edges: Single-channel image
        :param threshold: Threshold used for Hough lines
        :param lines_limit: If more than lines_limit lines are found raise an error. Used to avoid extremely heavy computation due to 4 vertices polygon combinations.
        :param hough_resolution: Resolution of the Hough transformation as a tuple (rho pixel res, theta degree res)
        :returns: A tuple containing 2 values: numpy array of corners and bi-dimensional list of segmented lines
        """
        lines = cv2.HoughLines(edges, *hough_res, threshold)
        if lines is None:
            return None
        lines = lines.squeeze(axis=1)

        # Differentiate lines in two clusters (horizontal and vertical)
        lines_groups = self.cluster_lines(lines)
        intersec = np.array(self.intersec_between_groups(lines_groups), dtype=np.int32).squeeze(axis=1)

        # Differentiate corners in four clusters (top-left, top-right, bottom-right, bottom-left)
        k = 4
        if (len(intersec) < k):
            raise CornersNotFound("Less than %d corners were found" % k)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(np.array(intersec, dtype=np.float32), k, None, criteria, 10, 
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
            if time.time_ns() - self.start_ns >= self.timeout_ns:
                self.timed_out = True
                break

        # Use as best corners the vertices of the polygon with the best score
        best = np.argmax(scores)
        return (np.array(comb[best]), lines_groups)

    def cluster_lines(self, lines):
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

    def intersec_between_groups(self, lines):
        """
        Compute intersection between two groups of lines
        :param lines: 2D list representing two groups of lines
        :returns : list of all the intersections
        """
        intersections = []
        for l1 in lines[0]:
            for l2 in lines[1]:
                intersections.append(self.intersection(l1, l2))

        return intersections

    def intersection(self, line1, line2):
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

