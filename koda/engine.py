from .corners.detector import CornersDetectorByEdges, CornersNotFound
from .ocr.tesseract import TesseractOCREngine
from .utils import *
import cv2

class DocumentNotFound(CornersNotFound):
    """
    Error occuring if no enough corners of a document were found
    """
    def __init__(self, message):
        super().__init__(message)

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
                r = np.array([[[w[1], w[2]], [w[3], w[4]]]], dtype=np.float32)
                r = cv2.perspectiveTransform(r, self.IM)[0]
                tl, br = (r[0][0], r[0][1]), (r[1][0], r[1][1])
                cv2.rectangle(overlay, tl, br, color, cv2.FILLED)
                res = cv2.addWeighted(overlay, alpha, self.img, 1-alpha, 0)

        return (res, res_warped)

class DetectionEngine:
    """
    Koda access point, detect a document from a given image through the Koda pipeline

    Attributes:
        cdetector: Corners detector. See /koda/corners/detector.py
        ocr: OCR engine. See /koda/ocr/tesseract.py
    """
    def __init__(self):
        self.cdetector = CornersDetectorByEdges()
        self.ocr = TesseractOCREngine()

    def detectDocument(self, img):
        """
        Find a document within the image. May raise DocumentNotFound.

        :param img: The image on which the search is applied
        :returns: A tuple containing a Document object and a dict of images (one per each pipeline step)
        """
        pipeline = Pipeline()

        # Detect corners
        try:
            corners = self.cdetector.find_corners(img)
        except CornersNotFound as e:
            raise DocumentNotFound(e.message)
        pipeline.next(self.cdetector.edges_img, label='edges')
        pipeline.next(self.cdetector.edges_img, hough_lines=self.cdetector.hough_lines)
        pipeline.next(img, corners=corners)

        # Warp image
        shape = (corners.max(axis=0)[0], corners.max(axis=0)[1])
        dst_corners = np.array([[0,0],[0, shape[1]],[shape[0], 0],[shape[0], shape[1]]])
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners.astype(np.float32))
        warped = cv2.warpPerspective(img, M, shape)
        pipeline.next(warped, label='warp')

        # Extract text
        try:
            words = self.ocr.extract_text(warped)
        except Exception:
            words = []

        return (Document(words, img, warped, M), pipeline.imgs)
