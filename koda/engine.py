from .corners.detector import CornersDetectorByEdges, CornersNotFound
from .ocr.tesseract import TesseractOCREngine
from .utils import *
import cv2

class DocumentNotFound(CornersNotFound):
    def __init__(self, message):
        super().__init__(message)

class Document:
    def __init__(self, img, words):
        self.words = words
        self.img = img

    def findWord(self, word):
        res = self.img.copy()
        for w in self.words:
            if (word.lower() in w[0].lower()):
                cv2.rectangle(res, (w[1], w[2]), (w[3], w[4]), (255,0,0), 2)
        return res

class DetectionEngine:
    def __init__(self):
        self.cdetector = CornersDetectorByEdges()
        self.ocr = TesseractOCREngine()

    def detectDocument(self, img):
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
        warped = corners_warp(img, corners)
        pipeline.next(warped, label='warp')

        # Color correction
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        color_corrected = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        pipeline.next(color_corrected, label='color_correction')

        # Extract text
        words = self.ocr.extract_text(warped)
        return (Document(warped, words), pipeline.imgs)
