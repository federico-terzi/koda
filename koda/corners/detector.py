import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import itertools
from scipy.signal import argrelextrema
from abc import ABC, abstractmethod
import time
from koda.edge.network import UNetEdgeDetector, TARGET_IMAGE_SIZE
from .utils import *

class CornersDetector(ABC):
    """
    Abstract class for corners detector of a document in an image
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
    Rely on a unet network to retrieve document edges and find Hough lines. Compute intersections as corners candidates.

    Useful properties
    - edges_img: Grayscale resized image of the detected edges
    - hough_lines: Get the best 4 Hough lines found on the edges_img
    """
    def __init__(self):
        super().__init__()
        self.noise_threshold = 80
        self.hough_threshold = 60
        self.hough_resolution = (1, np.pi/180)
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
        Iteratively find the best 4 Hough lines and adjust parameters if none were found.
        Compute corners as intersection of Hough lines.

        :param img: Input three-channel image
        :param iterations: Number of tries to find corners
        :returns: numpy 2D array representing the corners coordinates scaled to the original input image and ordered based on quadrants.
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
        lines = None
        for _ in range(iterations + 1):
            lines = self.find_hough_lines(img, self.hough_threshold, hough_res=self.hough_resolution)
            if lines is not None:
                break

            # Too few lines, try to decrease threshold
            self.hough_threshold = int(self.hough_threshold - (self.hough_threshold*0.1))

        if lines is None:
            raise CornersNotFound("No Hough lines were found")

        # Sort lines based on theta (parallel lines will be indexed near each other)
        lines = lines[lines[:,1].argsort()]

        # Compute corners 
        corners = [intersection(lines[0], lines[2]),
                intersection(lines[1], lines[3]),
                intersection(lines[2], lines[1]),
                intersection(lines[3], lines[0])]
        corners = cluster_points_quadrants(corners, (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE))

        self.hough_lines = lines
        return np.array([self.scale_corner(*c, TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE, w, h) for c in corners])

    def find_hough_lines(self, edges, threshold, hough_res=(1, np.pi/180)):
        """
        Search for Hough lines on the input image and find the best four by confidence.

        :param edges: The input single channel image
        :param threshold: Hough lines threshold
        :param hough_res: Hough lines resolution (default = (1, np.pi/180))
        :returns: Array of lines (two elements array) expressed using polar notation (rho, theta)
        """
        lines = cv2.HoughLines(edges, *hough_res, threshold)
        if lines is None:
            return None

        strong_lines = np.zeros([4,1,2])
        strong_index = 0
        for rho, theta in lines[:,0]:
            if rho < 0:
                rho *= -1
                theta -= np.pi

            closeness_rho = np.isclose(rho, strong_lines[0:strong_index,0,0], atol = 50)
            closeness_theta = np.isclose(theta, strong_lines[0:strong_index,0,1], atol = np.pi/36)
            closeness = np.all([closeness_rho, closeness_theta], axis=0)

            if not any(closeness) and strong_index < 4:
                strong_lines[strong_index] = np.array([rho, theta])
                strong_index += 1
        
        return strong_lines.squeeze(axis=1)

