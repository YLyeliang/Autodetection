from abc import ABCMeta,abstractmethod


class BaseBBoxCoder(metaclass=ABCMeta):
    """Base bounding box coder."""

    def __init__(self,**kwargs):
        pass

    @abstractmethod
    def encode(self,bboxes,gt_bboxes):
        """Encode delta between bboxes and ground truth boxes."""
        pass

    @abstractmethod
    def decode(self,bboxes,bboxes_pred):
        """Decode the predicted bboxes accoridng to prediction and base boxes."""
        pass