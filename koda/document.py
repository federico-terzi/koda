import cv2
import numpy as np

class Document:
    """
    Model class for a detected document

    Attributes:
        img: The warped image of the document
        words: List of all single word found in the document by OCR

    Methods:
        findWord: Search for a specified word in the document
    """
    def __init__(self, words, img, warped, M):
        self.words = words
        self.img = img
        self.warped = warped
        _, self.IM = cv2.invert(M)

    def findWord(self, word):
        """
        Find a word within the document

        :returns: A tuple of two images copy with the searched word highlighted. The first is a copy of the original image, the second a copy of the warped one.
        """
        overlay, overlay_warped = self.img.copy(), self.warped.copy()
        color = (0, 255, 255)
        alpha = 0.3
        for w in self.words:
            if (word.lower() in w[0].lower()):
                # Warped
                tl, br = (w[1], w[2]), (w[3], w[4])
                cv2.rectangle(overlay_warped, tl, br, color, cv2.FILLED)
                res_warped = cv2.addWeighted(overlay_warped, alpha, self.warped, 1-alpha, 0)

                # Original
                x0, y0, x1, y1 = w[1:5]
                pts = np.array([[[x0, y0], [x0, y1], [x1, y1], [x1, y0]]], dtype=np.float32)
                pts = cv2.perspectiveTransform(pts, self.IM)[0]
                cv2.fillConvexPoly(overlay, np.array(pts, dtype=np.int32), color, cv2.LINE_AA)
                res = cv2.addWeighted(overlay, alpha, self.img, 1-alpha, 0)

        return (res, res_warped)


