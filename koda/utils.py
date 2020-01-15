import math
import cv2
import numpy as np

class Pipeline:
    """
    Data structure to keep track of the Koda pipeline partial results
    
    Attributes:
        steps: List of strings storing the pipeline steps
        imgs: dict of images, one per each pipeline step
    """
    def __init__(self):
        self.steps = ['edges','hough_lines','corners','warp']
        self.imgs = dict()
        self.i = -1

    def next(self, img, **kwargs):
        self.i = (self.i + 1) % len(self.steps)
        step = self.steps[self.i]
        if step in ['edges', 'warp', 'color_correction']:
            self.set(img, kwargs['label'])
        elif step == 'hough_lines':
            self.hough_lines(img, kwargs['hough_lines'])
        else:
            self.corners(img, kwargs['corners'])

    def set(self, img, label):
        self.imgs[label] = img.copy()

    def corners(self, img, corners):
        corners_img = img.copy()
        csize = int(np.min(img.shape[0:2]) * 0.01)
        for x,y in corners:
            cv2.circle(corners_img, (x,y), csize, (255, 0, 0), -1)
        self.imgs["corners"] = corners_img

    def hough_lines(self, img, lines):
        lines_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        colors = [(255,0,0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for j, line in enumerate(lines):
            lines_img = draw_polar_line(lines_img, line, colors[j])
        self.imgs["hough_lines"] = lines_img

def draw_polar_line(img, line, color, thickness=1):
    """
    Draw a line, expressed in polar notaion (rho, theta) on a copy of the given image 
    
    :param img: Image which copy will be drawn on
    :param line: Array or tuple of two elements corresponding to rho and theta 
    :param color: BGR Color used to draw the line
    :param thickness: Thickness of the line (default = 1)
    :returns: A copy of the image with the line drawn
    """
    img_c = img.copy()
    distance_factor = max(img_c.shape)*2
    rho, theta = line
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + distance_factor*(-b)), int(y0 + distance_factor*(a)))
    pt2 = (int(x0 - distance_factor*(-b)), int(y0 - distance_factor*(a)))
    cv2.line(img_c, pt1, pt2, color, thickness, cv2.LINE_AA)
    return img_c
