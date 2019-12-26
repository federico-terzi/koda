import cv2
import numpy
from PIL import Image
import locale
locale.setlocale(locale.LC_ALL, 'C') # https://github.com/sirfz/tesserocr/issues/165
from tesserocr import PyTessBaseAPI, RIL, PSM

from .base import OCREngine

class TesseractOCREngine(OCREngine):
    """
    OCREngine implementation using Google Tesseract OCR combined 
    with the tesserocr bridge library.
    """

    def extract_text(self, image):
        """
        Given an arbitrary RGB image in numpy array format, return a list of all the words
        detected in the text, along with their bounding box coordinates.

        The returned list contains tuples in this format:
        (word, x1, y1, x2, y2)
        """

        # Convert the numpy array image to a Pillow-friendly format
        pillow_img = Image.fromarray(image)

        output = []

        # Open the Tesseract context, specifiying SPARSE_TEXT as an option (used to highlight 
        # single words, rather than lines of text )
        with PyTessBaseAPI(psm=PSM.SPARSE_TEXT) as api:
            api.SetImage(pillow_img)
            boxes = api.GetComponentImages(RIL.WORD, True)
            
            # Cycle through the boxes, populating the list
            for i, (im, box, _, _) in enumerate(boxes):
                api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
                ocrResult = api.GetUTF8Text()
                
                # Calculate the other bounding box coordinates
                x2, y2 = box['x']+box['w'], box['y']+box['h']

                entry = (ocrResult, box['x'], box['y'], x2, y2)
                output.append(entry)
        
        return output
