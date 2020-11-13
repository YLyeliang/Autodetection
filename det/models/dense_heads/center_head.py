from math import ceil, log
import torch
import torch.nn as nn
import torch.nn.functional as F
from mtcv.cnn import ConvModule, bias_init_with_prob
from mtcv.ops import batched_nms
import numpy as np

from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from ..utils import gaussian_radius, gen_gaussian_target


@HEADS.register_module()
class CenterHead(BaseDenseHead):
    """Head of CenterNet: Objects as Points

    Code is modified from the 'official github repo
    <https://github.com/xingyizhou/CenterNet>

    More details can be found in the 'paper
    <https://arxiv.org/abs/1904.07850>

    Args:
        num_classes (int): Number of categories excluding the background category.
        in_channel (int): Number of channels in the input feature map.
        num_feat_levels (int):

    """

    def __init__(self,
                 num_classes,
                 in_channel,
                 feat_channel=256,
                 dense_wh=False,
                 cat_spec_wh=False,
                 train_cfg=None,
                 test_cfg=None,
                 loss_hm=dict(
                     type='GaussianFocalLoss',
                     alpha=2.0,
                     gamma=4.0,
                     loss_weight=1),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_off=dict(type='L1Loss', loss_weight=1.0),
                 output_stride=4,
                 max_objects=128):
        super(CenterHead, self).__init__()

        self.num_classes = num_classes
        self.in_channel = in_channel
        self.feat_channel = feat_channel
        self.dense_wh = dense_wh
        self.cat_spec_wh = cat_spec_wh
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.wh_channel = 2 if not cat_spec_wh else 2 * num_classes

        self.loss_hm = build_loss(loss_hm)
        self.loss_wh = build_loss(loss_wh)
        self.loss_off = build_loss(loss_off)

        self.output_stride = output_stride
        self.max_objects = max_objects

    def _init_layers(self):
        self.feat_conv = nn.ModuleList()
        for i in range(3):
            self.feat_conv.append(nn.Conv2d(self.in_channel, self.feat_channel, kernel_size=3, padding=1, bias=True))
        self.relu = nn.ReLU(inplace=True)
        self.head_hm = nn.Conv2d(self.feat_channel, self.num_classes, kernel_size=1, stride=1, bias=True)
        self.head_wh = nn.Conv2d(self.feat_channel, self.wh_channel, kernel_size=1, stride=1, bias=True)
        self.head_offset = nn.Conv2d(self.feat_channel, 2, kernel_size=1, stride=1, bias=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # don't know why original author did this
        self.head_hm.bias.data.fill_(-2.19)

    def forward(self, x):
        hm_feat = self.feat_conv[0](x)
        wh_feat = self.feat_conv[1](x)
        off_feat = self.feat_conv[2](x)

        hm_score = self.head_hm(hm_feat)
        wh_pred = self.head_wh(wh_feat)
        off_pred = self.head_offset(off_feat)
        return hm_score, wh_pred, off_pred

    def loss(self,
             hm_score,
             wh_pred,
             off_pred,
             gt_bboxes,
             gt_labels,
             img_metas, ):
        """Compute losses of the head.

        Args:
            hm_score (Tensor): The heatmap score of centernet, with shape (N, num_classes, H, W)
            wh_pred (Tensor): The size prediction of centernet, with shape (N, 2, H, W)
            off_pred (Tensor): The offset prediction of centernet, with shape (N, 2, H, W)
            gt_bboxes (List(Tensor)): The ground truth bboxes in raw images fetched from dataset for each image,
            each one has shape [num_gt, 4].
            gt_labels (List(Tensor)): Class indices for each bounding box, each one with shape [num_gt,].
            img_metas (List(dict)): The image meta informations

        Returns:
        """
        feat_shape = hm_score.shape
        img_shape = img_metas['img_shape']
        targets = self.get_targets(gt_bboxes, gt_labels, feat_shape, img_shape)

        gt_hm = targets['heatmp']
        gt_wh = targets['wh']
        gt_off = targets['offset']
        gt_reg_mask = targets['reg_mask']
        gt_ind = targets['ind']

        wh_pred = self._transpose_and_gather_feat(wh_pred, gt_ind)
        reg_mask = gt_reg_mask.unsqueeze(2).expand_as(wh_pred).float()

        off_pred = self._transpose_and_gather_feat(off_pred, gt_ind)

        hm_loss = self.loss_hm(hm_score.sigmoid(), gt_hm, avg_factor=max(1, gt_hm.eq(1).sum()))
        wh_loss = self.loss_wh(wh_pred, gt_wh, reg_mask, avg_factor=max(1, reg_mask.sum()))
        off_loss = self.loss_off(off_pred, gt_off, reg_mask, avg_factor=max(1, reg_mask.sum()))

        loss_dict = dict(hm_loss=hm_loss, wh_loss=wh_loss, off_loss=off_loss)
        return loss_dict

    def get_targets(self,
                    gt_bboxes,
                    gt_labels,
                    feat_shape,
                    img_shape,
                    with_dense_wh=False,
                    ):
        """Compute the targets according to given ground truth bboxes and labels.
        The heatmap targets are computed using gaussian near center coordinates.
        the wh & offset are initialized with shape of [max_objects,2].
        L1 for reg and gaussian focal loss for heatmap class.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image, each
                has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box, each has
                shape (num_gt,).
            feat_shape (list[int]): Shape of output feature,
                [batch, channel, height, width].
            img_shape (list[int]): Shape of input image,
                [height, width, channel].

        Returns:
            target_result (dict): returning results contains all needed features.
            heatmap (Tensor): The heatmap of center point, according to gt bbox center, gaussian kernel,
            shape (batch_size, num_classes, height, width)
            wh (Tensor): The size features by gt bbox, with shape (n, max_objects, 2)
            offset (Tensor): The offset features by gt bbox, with shape (n, max_objects, 2)
            reg_mask (Tensor_: The regression mask, used to point out which object in target has assigned gt bbox.
            with shape (n, max_objects)
            ind (Tensor): The flattened index of center of gt bbox, counting from top-left and step row-by-row.

        """
        batch_size, _, height, width = feat_shape
        img_h, img_w = img_shape[:2]

        width_ratio = float(width / img_w)
        height_ratio = float(height / img_h)
        # initialize the gt target heatmap, wh, and reg
        gt_ind = gt_bboxes[-1].new_zeros([batch_size, self.max_objects], dtype=torch.int64)
        gt_reg_mask = gt_bboxes[-1].new_zeros([batch_size, self.max_objects], dtype=torch.uint8)
        # gt_cat_spec_wh = gt_bboxes[-1].new_zeros([batch_size, self.max_objects, self.num_classes * 2])
        # gt_cat_spec_mask = gt_bboxes[-1].new_zeros([batch_size, self.max_objects, self.num_classes * 2],
        #                                            dtype=torch.uint8)

        gt_heatmap = gt_bboxes[-1].new_zeros(
            [batch_size, self.num_classes, height, width])
        gt_offset = gt_bboxes[-1].new_zeros([batch_size, self.max_objects, 2])
        # gt_dense_wh = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
        gt_wh = gt_bboxes[-1].new_zeros([batch_size, self.max_objects, 2])

        for batch_id in range(batch_size):
            # Ground truth of center per image is a list of coord set
            for box_id in range(len(gt_labels[batch_id])):
                left, top, right, bottom = gt_bboxes[batch_id][box_id]
                center_x = (left + right) / 2.0
                center_y = (top + bottom) / 2.0
                label = gt_labels[batch_id][box_id]

                # Use coords in the feature level to generate ground truth
                scale_left = left * width_ratio
                scale_right = right * width_ratio
                scale_top = top * height_ratio
                scale_bottom = bottom * height_ratio
                scale_center_x = center_x * width_ratio
                scale_center_y = center_y * height_ratio

                # Int coords on feature map/ground truth tensor
                center_x_idx = int(scale_center_x)
                center_y_idx = int(scale_center_y)

                # Generate gaussian heatmap
                scale_box_width = ceil(scale_right - scale_left)
                scale_box_height = ceil(scale_bottom - scale_top)
                radius = gaussian_radius((scale_box_height, scale_box_width),
                                         min_overlap=0.7)
                radius = max(0, int(radius))
                gt_heatmap[batch_id, label] = gen_gaussian_target(gt_heatmap[batch_id, label],
                                                                  [center_x_idx, center_y_idx],
                                                                  radius)

                # Generate wh regression
                gt_wh[batch_id][box_id] = 1. * scale_box_width, 1. * scale_box_height

                # Generate offset
                gt_offset[batch_id][box_id] = [scale_center_x - center_x_idx, scale_center_y - center_y_idx]
                gt_reg_mask[batch_id][box_id] = 1
                gt_ind[batch_id][box_id] = center_y_idx * width + center_x_idx
                # gt_cat_spec_wh[batch_id, label * 2:label * 2 + 2] = gt_wh[batch_id][box_id]
                # gt_cat_spec_mask[batch_id, label * 2:label * 2 + 2] = 1

                # TODO: add this in later.
                # if with_dense_wh:

        targets_result = dict(
            heatmap=gt_heatmap,
            wh=gt_wh,
            offset=gt_offset,
            reg_mask=gt_reg_mask,
            ind=gt_ind
        )

        return targets_result

    def get_bboxes(self,
                   hm_score,
                   wh_pred,
                   off_pred,
                   img_metas,
                   rescale=False,
                   with_nms=True):
        """ Transform network output for a batch into bbox predictions.

        Args:
            hm_score (Tensor): output of head, with shape (n, num_classes, h, w)
            wh_pred (Tensor): output of head, (n ,2, h, w)
            off_pred (Tensor): output of head, (n, 2, h, w)
            img_metas (list(dict)): meta informations
            rescale (bool): Whether rescale to original size.
            with_nms (bool): Whether perform soft nms on final results.

        Returns:

        """
        assert hm_score.shape[0] == len(img_metas)
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(self._get_bboxes_single(hm_score[img_id:img_id + 1, :],
                                                       wh_pred[img_id:img_id + 1, :],
                                                       off_pred[img_id:img_id + 1, :],
                                                       img_metas[img_id],
                                                       rescale=rescale,
                                                       with_nms=with_nms))
        return result_list

    def _get_bboxes_single(self,
                           hm_score,
                           wh_pred,
                           off_pred,
                           img_meta,
                           rescale=False,
                           with_nms=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            hm_score (Tensor): output of head, with shape (1, num_classes, h ,w)
            wh_pred (Tensor): ...
            off_pred (Tensor): ...
            img_meta (dict): ...
            rescale (bool): ...
            with_nms (bool): ...

        Returns:
            bboxes and labels.
            The bboxes is a list contains all predicted bboxes with length of num_classes, each index is a result of related class.

        """
        if isinstance(img_meta, (list, tuple)):
            img_meta = img_meta[0]

        batch_bboxes, batch_scores, batch_clses = self.decode_heatmap(hm_score,
                                                                      wh_pred,
                                                                      off_pred,
                                                                      img_meta,
                                                                      k=self.test_cfg.center_topk,
                                                                      kernel=self.test_cfg.local_maximum_kernel)

        scale_factor = torch.Tensor(img_meta['scale_factor']).to(batch_bboxes.device)
        if rescale:
            batch_bboxes /= scale_factor
        bboxes = batch_bboxes.view([-1, 4])
        scores = batch_bboxes.view([-1, 1])
        clses = batch_clses.view([-1, 1])

        idx = scores.argsort(dim=0, descending=True)
        bboxes = bboxes[idx].view([-1, 4])
        scores = scores[idx].view(-1)
        clses = clses[idx].view(-1)

        detections = torch.cat([bboxes, scores.unsqueeze(-1)], -1)
        keepinds = (detections[:, -1] > -0.1)
        detections = detections[keepinds]
        labels = clses[keepinds]

        if with_nms:
            detections, labels = self._bboxes_nms(detections, labels, self.test_cfg)

        return detections, labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        out_bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:, -1], labels, cfg.nms_cfg)
        out_labels = labels[keep]

        if len(out_bboxes) > 0:
            idx = torch.argsort(out_bboxes[:, -1], descending=True)
            idx = idx[:cfg.max_per_img]
            out_bboxes = out_bboxes[idx]
            out_labels = out_labels[idx]
        return out_bboxes, out_labels

    def decode_heatmap(self,
                       heat,
                       wh,
                       off,
                       img_meta=None,
                       k=100,
                       kernel=3):
        """Transform outputs for a single batch item into a raw bbox predictions.

        Args:
            heat:
            wh:
            off:
            img_meta:
            k:
            kernel:
            distance_threshold:
            num_dets:

        Returns:

        """
        batch, _, height, width = heat.size()
        inp_h, inp_w, _ = img_meta['pad_shape']

        # perform nms on heatmaps
        heat = self._local_maximum(heat, kernel=kernel)

        scores, inds, clses, ys, xs = self._topk_centernet(heat, k=k)

        off = self._transpose_and_gather_feat(off, inds)
        off = off.view(batch, k, 2)
        xs = xs.view(batch, k, 1) + off[:, :, 0:1]
        ys = ys.view(batch, k, 1) + off[:, :, 1:2]

        wh = self._transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, k, 2)

        clses = clses.view(batch, k, 1).float()
        scores = scores.view(batch, k, 1)
        bboxes = torch.cat([(xs - wh[..., 0:1] / 2) * self.output_stride,
                            (ys - wh[..., 1:2] / 2) * self.output_stride,
                            (xs + wh[..., 0:1] / 2) * self.output_stride,
                            (ys + wh[..., 1:2] / 2) * self.output_stride], dim=2)

        return bboxes, scores, clses

    def _topk(self, scores, k=20):
        """Get top k positions from heatmap.

        Args:
            scores (Tensor): Target heatmap with shape
                [batch, num_classes, height, width].
            k (int): Target number. Default: 20.

        Returns:
            tuple[torch.Tensor]: Scores, indexes, categories and coords of
                topk keypoint. Containing following Tensors:

            - topk_scores (Tensor): Max scores of each topk keypoint.
            - topk_inds (Tensor): Indexes of each topk keypoint.
            - topk_clses (Tensor): Categories of each topk keypoint.
            - topk_ys (Tensor): Y-coord of each topk keypoint.
            - topk_xs (Tensor): X-coord of each topk keypoint.
        """
        batch, _, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
        topk_clses = (topk_inds / (height * width)).int()
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()
        return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

    def _topk_centernet(self, scores, k=20):
        """ The topk method in centernet"""
        batch, cat, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), k)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), k)
        topk_clses = (topk_ind / k).int()
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, k)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, k)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, k)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _local_maximum(self, heat, kernel=3):
        """Extract local maximum pixel with given kernal.

        Args:
            heat (Tensor): Target heatmap.
            kernel (int): Kernel size of max pooling. Default: 3.

        Returns:
            heat (Tensor): A heatmap where local maximum pixels maintain its
                own value and other positions are 0.
        """
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature according to index.

        Args:
            feat (Tensor): Target feature map.
            ind (Tensor): Target coord index.
            mask (Tensor | None): Mask of featuremap. Default: None.

        Returns:
            feat (Tensor): Gathered feature.
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _transpose_and_gather_feat(self, feat, ind):
        """Transpose and gather feature according to index.

        Args:
            feat (Tensor): Target feature map. With shape [N, C, H, W]
            ind (Tensor): Target coord index. [N, max_obj]

        Returns:
            feat (Tensor): Transposed and gathered feature.
        """
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat
