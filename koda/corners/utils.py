import cv2
import numpy as np
from collections import defaultdict

def cluster_points_quadrants(pts, img_shape):
    hor_sep = img_shape[0] // 2
    ver_sep = img_shape[1] // 2
    l = [list() for _ in range(4)]
    for p in pts:
        x, y = p
        if (x < hor_sep):
            if (y < ver_sep):
                l[0].append(p)
            else:
                l[1].append(p)
        else:
            if (y < ver_sep):
                l[2].append(p)
            else:
                l[3].append(p)
    return [item for sublist in l for item in sublist]

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
    except Exception:
        x0, y0 = np.linalg.lstsq(A, b)[0]
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]

