import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import itertools
from scipy.signal import argrelextrema
from abc import ABC, abstractmethod
import time
from koda.edge.network import UNetEdgeDetector, TARGET_IMAGE_SIZE
from .utilslines import *

class CornersDetector(ABC):
    """
    Abstrct class for corners detector of a document in an image
    """

    @abstractmethod
    def find_corners(self, img, iterations=3):
        """
        Find a document in the given image and returns its four corners.
        :param img: A three-channel image
        :param iterations: Number of tries to find the corners adjusting params for each iteration
        :returns: A numpy 2D array representing coordinate of the corners
        """
        pass

class CornersNotFound(RuntimeError):
    """
    Corners were not found in the image. Cause specified in the error message.
    """
    def __init__(self, message):
        super().__init__(message)

def millis():
    return int(round(time.time() * 1000))

class CornersDetectorByEdges(CornersDetector):
    """
    Rely on a U-Net network to retrieve document edges and find Hough lines. Compute intersections as corners candidates.
    Return the best corners found maximizing the quadrilater perimeter over the edges.

    Useful properties
    - timed_out: False if the found corners are optimal, True if the maximizing process was stopped by the timeout
    - hough_lines: Get the Hough lines grouped by angle referred to edges_img (note: img is heavily resized)
    - edges_img: Grayscale resized image of the detected edges
    """
    def __init__(self, timeout_ms=1000):
        """
        :param timeout_ms: Timeout specified in milliseconds after which the maximization stops
        """
        super().__init__()
        self.noise_threshold = 80
        self.hough_threshold = 60
        self.hough_resolution = (1, np.pi/36)
        self.start_ns = None
        self.timeout_ms = timeout_ms
        self.timed_out = False
        self.edge_detector = UNetEdgeDetector()
        self.edge_detector.load_model('koda/unet-70.h5')
        self.hough_lines = None
        self.edges_img = None

    def scale_corner(self, target_x, target_y, target_width, target_height, original_width, original_height):
        """
        Linear transform the given corner (target_x, target_y) found in an image of size (target_width, target_height)
        to match the corresponding point in the original image of size (original_width, original_height)
        """

        original_x = original_width * target_x / target_width
        original_y = original_height * target_y / target_height
        return (int(original_x), int(original_y))

    def find_corners(self, img, iterations=3):
        """
        Iteratively detect corners and adjust parameters if None corners were found.
        For corner detection detail see self.detect_corners

        :param img: Input three-channel image
        :param iterations: Number of tries to find corners
        :returns: numpy 2D array representing the corners coordinates
        """
        self.hough_lines = None
        self.timed_out = False
        self.start_ms = millis()

        h, w = img.shape[:-1]

        # Detect edge
        img = self.edge_detector.evaluate(img).astype(np.uint8)
        self.edges_img = img

        # Cleanup spourius pixel from edge detection model
        img[img < self.noise_threshold] = 0

        # Get corners, adjust params for each iterations tries specified
        e = None
        for _ in range(iterations + 1):
            try:
                corners, lines = self.detect_corners(img, self.hough_threshold, hough_res=self.hough_resolution)
                self.hough_lines = lines
            except CornersNotFound as e1:
                # Too few line, try to decrease threshold
                self.hough_threshold = int(self.hough_threshold - (self.hough_threshold*0.1))
                e = e1
                continue
            break

        if e is not None:
            raise e

        return np.array([self.scale_corner(*c, TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE, w, h) for c in corners])

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
            raise CornersNotFound("No Hough lines were found")
        lines = lines.squeeze(axis=1)

        # Differentiate lines in two clusters (horizontal and vertical)
        lines_groups = cluster_lines(lines)
        intersec = np.array(intersec_between_groups(lines_groups), dtype=np.int32).squeeze(axis=1)

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
            if millis() - self.start_ms >= self.timeout_ms:
                self.timed_out = True
                break

        # Use as best corners the vertices of the polygon with the best score
        best = np.argmax(scores)
        return (np.array(comb[best]), lines_groups)
