import math
import cv2
import numpy as np

class Pipeline:
    def __init__(self):
        self.steps = ['edges','hough_lines','corners','warp','color_correction']
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
        for x,y in corners:
            cv2.circle(corners_img, (x,y), 30, (255, 0, 0), 30)
        self.imgs["corners"] = corners_img

    def hough_lines(self, img, lines_groups):
        lines_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        colors = [(0,255,0),(0,0,255)]
        for j, lines in enumerate(lines_groups):
            lines_img = draw_polar_lines(lines_img, lines, colors[j])
        self.imgs["hough_lines"] = lines_img

def draw_polar_lines(img, lines, color, thickness=1):
    img_c = img.copy()
    distance_factor = max(img_c.shape)*2
    for rho, theta in lines:
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + distance_factor*(-b)), int(y0 + distance_factor*(a)))
        pt2 = (int(x0 - distance_factor*(-b)), int(y0 - distance_factor*(a)))
        cv2.line(img_c, pt1, pt2, color, thickness, cv2.LINE_AA)
    return img_c

def corners_warp(img, corners):
    shape = (corners.max(axis=0)[0], corners.max(axis=0)[1])
    dst_corners = np.array([[0,0],
                            [0, shape[1]],
                            [shape[0], 0],
                            [shape[0], shape[1]]])
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners.astype(np.float32))
    return cv2.warpPerspective(img, M, shape)
