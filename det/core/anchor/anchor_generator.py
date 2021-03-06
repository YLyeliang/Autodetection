import mtcv
import numpy as np
import torch
from torch.nn.modules.utils import _pair

from .builder import ANCHORE_GENERATORS


@ANCHORE_GENERATORS.register_module()
class AnchorGenerator(object):
    """
    Standard anchor generator for 2D anchor-based detectors.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels.
        ratios (list[float]): The list of ratios between the height and width
            of anchors in a single level.
        scales (list[int] | None): Anchor scales for anchors in a single level.
            It cannot be set at the same time if `octave_base_scale` and
            `scales_per_octave` are set.
        base_sizes (list[int] | None): The basic sizes
            of anchors in multiple levels.
            If None is given, strides will be used as base_sizes.
            (If strides are non square, the shortest stride is taken.)
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales. By default it is True in V2.0
        octave_base_scale (int): The base scale of octave.
        scales_per_octave (int): Number of scales for each octave.
            `octave_base_scale` and `scales_per_octave` are usually used in
            retinanet and the `scales` should be None when they are set.
        centers (list[tuple[float, float]] | None): The centers of the anchor
            relative to the feature grid center in multiple feature levels.
            By default it is set to be None and not used. If a list of tuple of
            float is given, they will be used to shift the centers of anchors.
        center_offset (float): The offset of center in propotion to anchors'
            width and height. By default it is 0 in V2.0.

    Examples:
        >>> from det.core import AnchorGenerator
        >>> self = AnchorGenerator([16], [1.], [1.], [9])
        >>> all_anchors = self.grid_anchors([(2, 2)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]])]
        >>> self = AnchorGenerator([16, 32], [1.], [1.], [9, 18])
        >>> all_anchors = self.grid_anchors([(2, 2), (1, 1)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]]), \
        tensor([[-9., -9., 9., 9.]])]
    """

    def __init__(self,
                 strides,
                 ratios,
                 scales=None,
                 base_sizes=None,
                 scale_major=True,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 centers=None,
                 center_offset=0.):
        # check center and center_offset
        if center_offset != 0:
            assert centers is None, 'center cannot be set when center_offset' \
                                    f'!=0, {centers} is given'
        if not (0 <= center_offset <= 1):
            raise ValueError('center_offset should be in range [0,1].'
                             f'{center_offset} is given.')
        if centers is not None:
            assert len(centers) == len(strides), 'The number of strides should be the same as centers, got' \
                                                 f'{strides} and {centers}'

        # calculate base sizes of anchors
        self.strides = [_pair(stride) for stride in strides]  # _pair is repeat(stride,2)
        self.base_sizes = [min(stride) for stride in self.strides] if base_sizes is None else base_sizes
        assert len(self.base_sizes) == len(self.strides), 'The number of strides should be the same as base sizes, got' \
                                                          f'{self.strides} and {self.base_sizes}'

        # calculate scales of anchors
        assert ((octave_base_scale is not None
                 and scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot be set at the same time.'
        if scales is not None:
            self.scales = torch.Tensor(scales)
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = np.array([2 ** (i / scales_per_octave) for i in range(scales_per_octave)])
            # above e.g. scales_per_octave=4, then 2^0 2^(1/4) 2^(1/2) 2^(3/4)
            scales = octave_base_scale * octave_scales
            self.scales = torch.Tensor(scales)
        else:
            raise ValueError('Either scales or octave_base_scale with'
                             'scales_per_octave should be set')

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        """list[int] total number of base anchors in a feature grid"""
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied."""
        return len(self.strides)

    def gen_base_anchors(self):
        """
        Generate base anchors.
        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
            feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    center=center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        """
        Generate base anchors of a single level.
        Args:
            base_size: (int | float) Basic size of an anchor.
            scales: (torch.Tensor) Scales of the anchor.
            ratios: (torch.Tensor) ratios between the height & width of anchors in single level.
            center: (tuple[float], optional) The center of the base anchor related to a single feature grid.
                Default to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature maps.

        """
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            # w_ratios is list. w_ratios[:,None] add a new dim, then become [n,1], scales [1,n] -> [n,n]
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        # use float anchor and the anchor's center is aligned with the pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        """
        Generate mesh grid of x and y.
        Args:
            x (torch.Tensor): Grids of x dimension.
            y (torch.Tensor): Grids of y dimension.
            row_major (bool, optional): Whether to return y grids first.
                Defaults to True.

        Returns:
            tuple[torch.Tensor]: The mesh grids of x and y.
        """
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_sizes, device='cuda'):
        """
        Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
            device (str): Device where the anchors will be put on.

        Return:
            list[torch.Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature lavel, \
                num_base_anchors is the number of anchors for that level.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                self.base_anchors[i].to(device),
                featmap_sizes[i],
                self.strides[i],
                device=device)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16),
                                  device='cuda'):
        """
        Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_anchors``.

        Args:
            base_anchors (torch.Tensor): The base anchors of a feature grid.
            featmap_size (tuple[int]): Size of the feature maps.
            stride (tuple[int], optional): Stride of the feature map.
                Defaults to (16, 16).
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride[0]
        shift_y = torch.arange(0, feat_h, device=device) * stride[1]
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1,A,4) to k shifts (k,1,4) to get
        # shifted anchors (k,A,4), reshape to (k*A,4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0,0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
        """
        Generate valid flags of anchors in multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels.
            pad_shape (tuple): The padded shape of the image.
            device (str): Device where the anchors will be put on.

        Return:
            list(torch.Tensor): Valid flags of anchors in multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            # min(feature map size, padded image size / stride)
            valid_feat_h = min(int(np.ceil(h / anchor_stride[0])), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride[1])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  self.num_base_anchors[i],
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size,
                                 num_base_anchors,
                                 device='cuda'):
        """
        Generate the valid flags of anchor in a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps.
            valid_size (tuple[int]): The valid size of the feature maps.
            num_base_anchors (int): The number of base anchors.
            device (str, optional): Device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each anchor in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(valid.size(0), num_base_anchors).contiguous().view(-1)
        return valid

    def __repr__(self):
        """str: a string that describes the module"""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides},\n'
        repr_str += f'{indent_str}ratios={self.ratios},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}base_sizes={self.base_sizes},\n'
        repr_str += f'{indent_str}scale_major={self.scale_major},\n'
        repr_str += f'{indent_str}octave_base_scale='
        repr_str += f'{self.octave_base_scale},\n'
        repr_str += f'{indent_str}scales_per_octave='
        repr_str += f'{self.scales_per_octave},\n'
        repr_str += f'{indent_str}num_levels={self.num_levels}\n'
        repr_str += f'{indent_str}centers={self.centers},\n'
        repr_str += f'{indent_str}center_offset={self.center_offset})'
        return repr_str


@ANCHORE_GENERATORS.register_module()
class SSDAnchorGenerator(AnchorGenerator):
    """
    Anchor generator for SSD.

    Args:
        strides (list[int]  | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels.
        ratios (list[float]): The list of ratios between the height and width
            of anchors in a single level.
        basesize_ratio_range (tuple(float)): Ratio range of anchors.
        input_size (int): Size of feature map, 300 for SSD300,
            512 for SSD512.
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales. It is always set to be False in SSD.
    """

    def __init__(self,
                 strides,
                 ratios,
                 basesize_ratio_range,
                 input_size=300,
                 scale_major=True):
        assert len(strides) == len(ratios)
        assert mtcv.is_tuple_of(basesize_ratio_range, float)

        self.strides = [_pair(stride) for stride in strides]
        self.input_size = input_size
        self.centers = [(stride[0] / 2., stride[1] / 2.) for stride in self.strides]
        self.basesize_ratio_range = basesize_ratio_range

        # calculate anchor ratios and scales
        min_ratio, max_ratio = basesize_ratio_range
        min_ratio = int(min_ratio * 100)
        max_ratio = int(max_ratio * 100)
        step = int(np.floor(max_ratio - min_ratio) / (self.num_levels - 2))
        min_sizes = []
        max_sizes = []
        # TODO: Figure out the process
        for ratio in range(int(min_ratio), int(max_ratio) + 1, step):
            min_sizes.append(int(self.input_size * ratio / 100))
            max_sizes.append(int(self.input_size * (ratio + step) / 100))
        if self.input_size == 300:
            if basesize_ratio_range[0] == 0.15:  # SSD300 COCO
                min_sizes.insert(0, int(self.input_size * 7 / 100))
                max_sizes.insert(0, int(self.input_size * 15 / 100))
            elif basesize_ratio_range[0] == 0.2:  # SSD300 VOC
                min_sizes.insert(0, int(self.input_size * 10 / 100))
                max_sizes.insert(0, int(self.input_size * 20 / 100))
            else:
                raise ValueError(
                    'basesize_ratio_range[0] should be either 0.15 or 0.2 when input_size is 300, got'
                    f'{basesize_ratio_range[0]}.')
        elif self.input_size == 512:
            if basesize_ratio_range[0] == 0.1:  # SSD512 COCO
                min_sizes.insert(0, int(self.input_size * 4 / 100))
                max_sizes.insert(0, int(self.input_size * 10 / 100))
            elif basesize_ratio_range[0] == 0.15:  # SSD512 VOC
                min_sizes.insert(0, int(self.input_size * 7 / 100))
                max_sizes.insert(0, int(self.input_size * 15 / 100))
            else:
                raise ValueError(
                    'basesize_ratio_range[0] should be either 0.1 or 0.2 when input_size is 512, got'
                    f'{basesize_ratio_range[0]}.')
        else:
            raise ValueError(f'Only support 300 or 512 in SSDAnchorGenrator, got {self.input_size}')

        anchor_ratios = []
        anchor_scales = []
        for k in range(len(self.strides)):
            scales = [1., np.sqrt(max_sizes[k] / min_sizes[k])]
            anchor_ratio = [1.]
            for r in ratios[k]:
                anchor_ratio += [1 / r, r]  # 4 or 6 ratio
            anchor_ratios.append(torch.Tensor(anchor_ratio))
            anchor_scales.append(torch.Tensor(scales))

        self.base_sizes = min_sizes
        self.scales = anchor_scales
        self.ratios = anchor_ratios
        self.scale_major = scale_major
        self.center_offset = 0
        self.base_anchors = self.gen_base_anchors()

    def gen_base_anchors(self):
        """
        Generate base anchors.

        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            base_anchors = self.gen_single_level_base_anchors(
                base_size,
                scales=self.scales[i],
                ratios=self.ratios,
                center=self.centers[i])
            indices = list(range(len(self.ratios[i])))
            indices.insert(1, len(indices))
            base_anchors = torch.index_select(base_anchors, 0, torch.LongTensor(indices))
            multi_level_base_anchors.append(base_anchors)

        return multi_level_base_anchors

    def __repr__(self):
        """str: a string that describes the module"""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}scale_major={self.scale_major},\n'
        repr_str += f'{indent_str}input_size={self.input_size},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}ratios={self.ratios},\n'
        repr_str += f'{indent_str}num_levels={self.num_levels},\n'
        repr_str += f'{indent_str}base_sizes={self.base_sizes},\n'
        repr_str += f'{indent_str}basesize_ratio_range='
        repr_str += f'{self.basesize_ratio_range})'
        return repr_str


@ANCHORE_GENERATORS.register_module()
class YOLOAnchorGenerator(object):

    def __init__(self,
                 strides=[8, 16, 32],
                 anchors=[[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243],
                          [459, 401]],
                 device=None):
        self.strides = strides
        self.anchors = anchors
        self.device = device
        self.num_level = len(strides)
        self.anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    def grid_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """
        Generate yolo format anchors.
        Args:
            featmap_sizes:

        Returns:

        """
        self.batch_size = len(img_metas)
        self.device = device
        self.masked_anchors, self.grid_x, self.grid_y, self.ref_anchors, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(self.num_level):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anchor_masks[i]], dtype=np.float32)
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)
            # calculate pred - xywh obj cls
            featmap_size = featmap_sizes[i]
            grid_x = torch.arange(featmap_size, dtype=torch.float).repeat(self.batch_size, 3, featmap_size, 1).to(
                self.device)
            grid_y = torch.arange(featmap_size, dtype=torch.float).repeat(self.batch_size, 3, featmap_size, 1).permute(
                0, 1, 3, 2).to(self.device)
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(self.batch_size, featmap_size, featmap_size,
                                                                     1).permute(0, 3, 1, 2).to(self.device)
            anchow_h = torch.from_numpy(masked_anchors[:, 1]).repeat(self.batch_size, featmap_size, featmap_size,
                                                                     1).permute(0, 3, 1, 2).to(self.device)
            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchow_h)
        return (self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h)
