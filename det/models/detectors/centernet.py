from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class CenterNet(SingleStageDetector):
    """
    Implementation of Yolov4.
    """

    def __init__(self,
                 backbone,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CenterNet, self).__init__(backbone, bbox_head, train_cfg, test_cfg, pretrained)
