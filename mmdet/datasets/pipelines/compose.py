import collections

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    # 绘制全图
    # import matplotlib.pyplot as plt
    # plt.imshow(data['img'])
    # ax = plt.gca()
    # i = 79
    # xy = data['ann_info']['roof_masks'][i][0].copy()
    # xy = [[xy[2 * i], xy[2 * i + 1], ] for i in range(int(len(xy) / 2))]
    # ax.add_patch(plt.Polygon(xy=xy))
    # x, y, width, height = data['ann_info']['bboxes'][i][0], data['ann_info']['bboxes'][i][1], \
    #                       data['ann_info']['bboxes'][i][2] - data['ann_info']['bboxes'][i][0], \
    #                       data['ann_info']['bboxes'][i][3] - data['ann_info']['bboxes'][i][1]
    # ax.add_patch(
    #     plt.Rectangle((x, y), width, height, color="blue",
    #                   fill=False,
    #                   linewidth=1))
    # plt.show()

    # 绘制bbox内的
    # import matplotlib.pyplot as plt
    # import numpy as np
    # i = 84
    # left = int(data['ann_info']['bboxes'][i][0])
    # top = int(data['ann_info']['bboxes'][i][1])
    # right = int(data['ann_info']['bboxes'][i][2])
    # bottom = int(data['ann_info']['bboxes'][i][3])
    # plt.imshow(data['img'][top:bottom, left:right])
    # ax = plt.gca()
    # xy = data['ann_info']['roof_masks'][i][0].copy()
    # xy = [[xy[2 * i] - left, xy[2 * i + 1] - top, ] for i in range(int(len(xy) / 2))]
    # ax.add_patch(plt.Polygon(xy=xy, fill=False))
    # xy = np.array(xy)
    # proc = np.dot(xy, data['ann_info']['offsets'][i])
    # min_point = xy[np.argmin(proc)]
    # plt.scatter(min_point[0], min_point[1])
    # plt.plot([min_point[0], min_point[0] - data['ann_info
    # ']['offsets'][i][0]],
    #          [min_point[1], min_point[1] - data['ann_info']['offsets'][i][1]])
    # # ax.fill(xy)
    # plt.show()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
