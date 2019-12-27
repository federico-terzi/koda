from .engine import *

deng = DetectionEngine()

def load(img):
    """
    Load the given image in the Koda engine and starts the document detection pipeline.

    :returns: A tuple (Document, images). 'Document' is an object containing the detected document and its text. 'images' is a dict of images showing partials results of the Koda pipeline.
    """
    return deng.detectDocument(img)
