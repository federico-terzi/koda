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
    def __init__(self, img, words):
        self.words = words
        self.img = img

    def findWord(self, word):
        """
        Find a word within the document
        
        :returns: A copy of the document image with bounding box drawn around the searched word
        """
        res = self.img.copy()
        for w in self.words:
            if (word.lower() in w[0].lower()):
                cv2.rectangle(res, (w[1], w[2]), (w[3], w[4]), (255,0,0), 2)
        return res

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
        warped = corners_warp(img, corners)
        pipeline.next(warped, label='warp')

        # Extract text
        try:
            words = self.ocr.extract_text(warped)
        except Exception:
            words = []

        return (Document(warped, words), pipeline.imgs)
