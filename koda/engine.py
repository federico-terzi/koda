from .corners.detector import CornersDetectorByEdges, CornersNotFound
from .ocr.tesseract import TesseractOCREngine
from .utils import *
from .document import *
import cv2

class DocumentNotFound(CornersNotFound):
    """
    Error occuring if no enough corners of a document were found
    """
    def __init__(self, message):
        super().__init__(message)

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
        tl, tr, bl, br = corners
        h1, h2 = bl[0] - tl[0], br[0] - tr[0]
        w1, w2 = tr[1] - tl[1], br[1] - bl[1]
        shape = (int(np.mean([h1, h2])), int(np.mean([w1, w2])))
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
