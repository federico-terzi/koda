from abc import ABC, abstractmethod

class EdgeDetector(ABC):
    """
    Basic interface for the Edge detection component.
    """

    @abstractmethod
    def evaluate(self, image):
        """
        Given an arbitrary RGB image in numpy array format, return a 256x256 mask of the document edges.
        """
        pass