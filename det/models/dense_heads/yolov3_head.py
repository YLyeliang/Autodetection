import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mtcv.cnn import ConvModule
from mtcv.cnn.weight_init import normal_init, bias_init_with_prob

from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from det.core import multi_apply, images_to_levels, unmap, anchor_inside_flags, build_anchor_generator


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True, GIoU=False, DIoU=False, CIoU=False):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    https://github.com/ultralytics/yolov3/blob/eca5b9c1d36e4f73bf2f94e141d864f1c2739e23/utils/utils.py#L262-L282
    """
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        # intersection top left
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # intersection bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        con_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, 0] + bboxes_a[:, None, 2]) - (bboxes_b[:, 0] + bboxes_b[:, 2])) ** 2 / 4 + (
                (bboxes_a[:, None, 1] + bboxes_a[:, None, 3]) - (bboxes_b[:, 1] + bboxes_b[:, 3])) ** 2 / 4

        w1 = bboxes_a[:, 2] - bboxes_a[:, 0]
        h1 = bboxes_a[:, 3] - bboxes_a[:, 1]
        w2 = bboxes_b[:, 2] - bboxes_b[:, 0]
        h2 = bboxes_b[:, 3] - bboxes_b[:, 1]

        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # intersection top left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # intersection bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # convex (smallest enclosing box) top left and bottom right
        con_tl = torch.min((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        con_br = torch.max((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                           (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        # centerpoint distance squared
        rho2 = ((bboxes_a[:, None, :2] - bboxes_b[:, :2]) ** 2 / 4).sum(dim=-1)

        w1 = bboxes_a[:, 2]
        h1 = bboxes_a[:, 3]
        w2 = bboxes_b[:, 2]
        h2 = bboxes_b[:, 3]

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_u = area_a[:, None] + area_b - area_i
    iou = area_i / area_u

    if GIoU or DIoU or CIoU:
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            area_c = torch.prod(con_br - con_tl, 2)  # convex area
            return iou - (area_c - area_u) / area_c  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    return iou


@HEADS.register_module()
class YoloHead(BaseDenseHead):
    r"""
    Yolov3 head used to complete the YoloV4 detector.

    Unlike RetinaNet that using shared conv for multi-level features to predict confidence and location.
    yolov3 use separate 3x3-1x1 conv for each level features.

    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='LeakyReLU'),
                 anchor_generator=dict(
                     type='YOLOAnchorGenerator',
                     strides=[8, 16, 32],
                     anchors=[[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243],
                              [459, 401]],
                     device='cuda', ),
                 **kwargs):
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_level = len(in_channels)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.num_anchors = 3

        self.anchor_generator = build_anchor_generator(anchor_generator)
        super(YoloHead, self).__init__()

    def _init_layers(self):
        """Initialize the layers of the head."""
        # self.conv1 = ConvModule(self.in_channels, self.feat_channels, kernel_size=3, padding=1, bias=False,
        #                         conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        self.conv_lists = nn.ModuleList()
        self.conv_pred = nn.ModuleList()
        for i in range(self.num_level):
            conv1 = ConvModule(self.in_channels, self.feat_channels * 2, kernel_size=3, padding=1, bias=False,
                               conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            conv2 = ConvModule(self.feat_channels * 2, self.feat_channels, kernel_size=1, bias=False,
                               conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            self.conv_lists.append(nn.ModuleList([conv1, conv2]))
            self.conv_pred.append(
                nn.Conv2d(self.feat_channels, self.num_anchors * (4 + 1 + self.num_classes), kernel_size=1))

    def init_weights(self):
        """Initialize weights of the head."""
        for conv_list in self.conv_lists:
            for m in conv_list:
                normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        for m in self.conv_pred:
            normal_init(m, std=0.01, bias=bias_cls)

    def forward(self, feats):
        """
        Forward features from network. i.e, PAFPN.
        Unlike RetinaNet,
        Args:
            feats: (tuple[Tensor]): Features with multi-level from network, each is 4-D tensor.

        Returns:
            tuple: A tuple of outputs with each one has shape [N, (4+1+num_classes)* num_anchors, feat_size,feat_size]
        """
        outs = []
        for i, feat in enumerate(feats):
            feat = self.conv_lists[i][0](feat)
            feat = self.conv_lists[i][1](feat)
            out = self.conv_pred[i](feat)
            outs.append(out)
        return tuple([outs])

    def get_bboxes(self, **kwargs):
        pass


    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """
        Get anchors accordings to the feature map sizes.
        Args:
            featmap_size: (list[tuple]): Multi-level feature map sizes.
            img_metas: (list[dict]): Image meta info.
            device: (torch.device |str): Device for returned tensors.

        Returns:
            list:
                multi_level_anchors: (masked_anchors, ref_anchors, grid_x, grid_y, anchor_w, anchor_h)
        """

        # feature map sizes of all images are the same, so only one time are calculated.
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, img_metas, device)
        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = multi_level_anchors

    def get_targets(self,
                    predictions,
                    gt_bboxes,
                    gt_labels,
                    img_metas,
                    featmap_size,
                    level_id,
                    ignore_thr=0.4,
                    device='cuda'):
        """
        Get target anchors.
        Args:
            predictions:
            gt_bboxes:
            img_metas:
            featmap_size:

        Returns:

        """
        batch_size = len(img_metas)
        target_mask = torch.zeros(batch_size, self.num_anchors, featmap_size, featmap_size, 4 + self.num_classes).to(
            device=device)
        obj_mask = torch.ones(batch_size, self.num_anchors, featmap_size, featmap_size).to(device=device)
        target_scale = torch.zeros(batch_size, self.num_anchors, featmap_size, featmap_size, 2).to(device=device)
        target = torch.zeros(batch_size, self.num_anchors, featmap_size, featmap_size, 5 + self.num_classes).to(
            device=device)

        for b in range(batch_size):
            obj_mask[b], target_mask[b], target_scale[b], target[b] = self.get_target_single(
                predictions[b],
                gt_bboxes[b],
                gt_labels[b],
                featmap_size,
                level_id,
                anchor_mask=self.anchor_generator.anchor_masks,
                ignore_thr=ignore_thr,
                device=device)
        return obj_mask, target_mask, target_scale, target

    def get_target_single(self,
                          pred,
                          gt_bboxes,
                          gt_labels,
                          featmap_size,
                          level_id,
                          anchor_mask,
                          ignore_thr=0.4,
                          device='cuda'):
        """
        Compute regression and classification targets for anchors in a single image with single level.
        Args:
            pred (Tensor): prediction from network, with shape (num_anchors, featmapsize, featmapsize, 4)
            gt_bboxes (Tensor): Ground truth bboxes of the image, shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be ignored, (num_ignored_gts,4).
            gt_labels (Tensor): Ground truth labels of each box, shape (num_gts,).
            batch_size:
            featmap_size:
            level_id:
            ignore_thre:
            device:

        Returns:

        """
        stride = self.anchor_generator.strides[level_id]
        target_mask = torch.zeros(self.num_anchors, featmap_size, featmap_size, 4 + self.num_classes).to(device=device)
        obj_mask = torch.ones(self.num_anchors, featmap_size, featmap_size).to(device=device)
        target_scale = torch.zeros(self.num_anchors, featmap_size, featmap_size, 2).to(device=device)
        target = torch.zeros(self.num_anchors, featmap_size, featmap_size, 5 + self.num_classes).to(device=device)

        n = gt_labels.shape[0]

        # with shape (num_gt,)
        gt_x = (gt_bboxes[:, 2] + gt_bboxes[:, 0]) / (stride * 2)
        gt_y = (gt_bboxes[:, 3] + gt_bboxes[:, 1]) / (stride * 2)
        gt_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) / stride
        gt_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]) / stride
        gt_x_shift = gt_x.to(torch.int16).cpu().numpy()
        gt_y_shift = gt_y.to(torch.int16).cpu().numpy()

        gt_bbox = torch.zeros(n, 4).to(device)
        gt_bbox[:n, 2] = gt_w[:n]
        gt_bbox[:n, 3] = gt_h[:n]

        # calculate iou between gt and reference anchors
        # both gt and ref_anchors here are (0, 0, w, h), and get ious with shape (num_gt,num_ref_anchor=9)
        anchor_ious = bboxes_iou(gt_bbox.cpu(), self.ref_anchors[level_id], CIoU=True)

        # find argmax of ref_anchor that gt have best iou.
        best_n_all = anchor_ious.argmax(dim=1)
        best_n = best_n_all % 3
        best_n_mask = ((best_n_all == anchor_mask[level_id][0]) |
                       (best_n_all == anchor_mask[level_id][1]) |
                       (best_n_all == anchor_mask[level_id][2]))
        if sum(best_n_mask) == 0:
            return obj_mask, target_mask, target_scale, target
        gt_bbox[:n, 0] = gt_x[:n]
        gt_bbox[:n, 1] = gt_y[:n]

        pred_ious = bboxes_iou(pred.view(-1, 4), gt_bbox, xyxy=False)
        pred_best_iou, _ = pred_ious.max(dim=1)
        pred_best_iou = (pred_best_iou > ignore_thr)
        pred_best_iou = pred_best_iou.view(pred.shape[:3])  # (num_anchors,fsize,fsize)
        # set mask to zero (ignore) if pred matches gt
        obj_mask = ~pred_best_iou

        for ti in range(best_n.shape[0]):
            if best_n_mask[ti] == 1:
                i, j = gt_x_shift[ti], gt_y_shift[ti]
                a = best_n[ti]
                obj_mask[a, j, i] = 1
                target_mask[a, j, i, :] = 1
                # e.g. 1.4 (float) int16:1 target_x: 1.4-1=.04
                target[a, j, i, 0] = gt_x[ti] - gt_x[ti].to(torch.int16).to(torch.float)
                target[a, j, i, 1] = gt_y[ti] - gt_y[ti].to(torch.int16).to(torch.float)
                # w: log(gt_w / anchor_w) . since b_w = p_w * e^{t_w}, -> t_w = log(b_w / p_w)
                target[a, j, i, 2] = torch.log(
                    gt_w[ti] / torch.Tensor(self.masked_anchors[level_id])[best_n[ti], 0] + 1e-7)
                target[a, j, i, 3] = torch.log(
                    gt_h[ti] / torch.Tensor(self.masked_anchors[level_id])[best_n[ti], 1] + 1e-7)
                target[a, j, i, 4] = 1
                target[a, j, i, 5 + gt_labels[ti].to(torch.int16).cpu().numpy()] = 1
                # sqrt(2-w*h/fsize/fsize)
                target_scale[a, j, i, :] = torch.sqrt(2 - gt_w[ti] * gt_h[ti] / featmap_size / featmap_size)
        return obj_mask, target_mask, target_scale, target

    def loss(self,
             predictions,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """
        Calculate the loss of yolo.
        Args:
            predictions (tuple(Tensor)): the multi-level predictions from network, each level has shape (N,(5+num_classes)*num_anchors, feat_size, feat_size)
            gt_bboxes (list(Tensor)): Ground truth bboxes for each image with shape (num_gts, 4) in [xmin,ymin,xmax,ymax] format.
            gt_labels (list(Tensor)) : Class indices corresponding to each box
            img_metas (list(dict))  : Meta information of each image, e.g., image size, scaling factor, etc.
            gt_bboxes_ignore:

        Returns:

        """
        featmap_sizes = [pred.size()[-1] for pred in predictions]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        batch_size = len(img_metas)
        device = predictions[0].device
        self.get_anchors(
            featmap_sizes, img_metas, device=device)

        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
        for level_id, output in enumerate(predictions):
            featmap_size = output.shape[2]
            anchor_channel = 5 + self.num_classes

            output = output.view(batch_size, self.num_anchors, anchor_channel, featmap_size, featmap_size)
            output = output.permute(0, 1, 3, 4, 2)

            # decode pred bbox
            output[..., np.r_[:2, 4:anchor_channel]] = torch.sigmoid(output[..., np.r_[:2, 4:anchor_channel]])

            pred = output[..., :4].clone()
            pred[..., 0] += self.grid_x[level_id]
            pred[..., 1] += self.grid_y[level_id]
            pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[level_id]
            pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[level_id]

            obj_mask, target_mask, target_scale, target = self.get_targets(pred, gt_bboxes, gt_labels, img_metas,
                                                                           featmap_size, level_id, device=device)

            # loss calculation
            output[..., 4] *= obj_mask
            output[..., np.r_[0:4, 5:anchor_channel]] *= target_mask
            output[..., 2:4] *= target_scale

            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:anchor_channel]] *= target_mask
            target[..., 2:4] *= target_scale

            loss_xy += F.binary_cross_entropy(output[..., :2], target[..., :2], weight=target_scale * target_scale,reduction='sum')
            loss_obj += F.binary_cross_entropy(output[..., 4], target[..., 4],reduction='sum')
            loss_cls += F.binary_cross_entropy(output[..., 5:], target[..., 5:],reduction='sum')

            loss_wh += F.mse_loss(output[..., 2:4], target[..., 2:4],reduction='sum') / 2
            loss_l2 += F.mse_loss(output, target,reduction='sum')

        loss = loss_xy + loss_wh + loss_obj + loss_cls
        return dict(loss=loss, loss_xy=loss_xy, loss_wh=loss_wh, loss_obj=loss_obj, loss_cls=loss_cls, loss_l2=loss_l2)
