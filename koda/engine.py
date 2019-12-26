from .corners.detector import CornersDetectorByEdges, CornersNotFound
from .ocr.tesseract import TesseractOCREngine
from .utils import *

class DocumentNotFound(CornersNotFound):
    def __init__(self, message):
        super().__init__(message)

class Document:
    def __init__(self, words):
        self.word = words

    def findText(text):
        res = []
        for w in self.words:
            if (w[0] == text):
                res.append(w)
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
        # TODO
        color_corrected = warped
        pipeline.next(color_corrected, label='color_correction')

        # Extract text
        #words = self.ocr.extract_text(color_corrected)
        words = []
        return (Document(words), pipeline.imgs)
