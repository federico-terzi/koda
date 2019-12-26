from .engine import *

deng = DetectionEngine()

def load(img):
    return deng.detectDocument(img)
