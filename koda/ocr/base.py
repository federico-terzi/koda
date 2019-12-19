from abc import ABC, abstractmethod

class OCREngine(ABC):
    """
    Basic interface for the OCR detection component
    """

    @abstractmethod
    def extract_text(self, image):
        """
        Given an arbitrary RGB image in numpy array format, return a list of all the words
        detected in the text, along with their bounding box coordinates.

        The returned list contains tuples in this format:
        (word, x1, y1, x2, y2)
        """
        pass