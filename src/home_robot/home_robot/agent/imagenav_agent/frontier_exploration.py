# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import skimage.morphology
import torch
import torch.nn as nn

from home_robot.mapping.semantic.constants import MapConstants as MC
from home_robot.utils.morphology import binary_dilation, binary_erosion

# 用于实现前沿探索策略。前沿探索是一种常用于自动化地图绘制和机器人导航的策略，目的是找到并探索尚未探索的区域边缘。
class FrontierExplorationPolicy(nn.Module):
    """
    Frontier exploration: select high-level exploration goals of the closest
    unexplored region.
    """
    # 初始化 FrontierExplorationPolicy 类的实例。设置用于地图膨胀和边缘选择的核（kernel）
    def __init__(self) -> None:
        super().__init__()

        self.dilate_explored_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(10))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )
        self.select_border_kernel = nn.Parameter(
            torch.from_numpy(skimage.morphology.disk(1))
            .unsqueeze(0)
            .unsqueeze(0)
            .float(),
            requires_grad=False,
        )

    @property
    # 提供更新探索目标的步数间隔。
    def goal_update_steps(self) -> int:
        return 1
        
    # 输入语义地图特征（map_features），该方法执行前沿探索策略的核心算法。
    # 该算法首先确定未探索的区域，然后对探索过的区域进行膨胀操作，以此来找到探索与未探索区域的边界（即“前沿”）。
    # 最后，它返回一个二值地图（goal_map），标识了应该作为探索目标的前沿区域。
    def forward(self, map_features: np.ndarray) -> np.ndarray:
        """
        Arguments:
            map_features: semantic map features of shape
             (batch_size, 8 + num_sem_categories, M, M)

        Returns:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
        """
        # Select unexplored area
        frontier_map = (map_features[:, [MC.EXPLORED_MAP], :, :] == 0).float()

        # Dilate explored area
        frontier_map = binary_erosion(frontier_map, self.dilate_explored_kernel)

        # Select the frontier
        frontier_map = (
            binary_dilation(frontier_map, self.select_border_kernel) - frontier_map
        )
        return frontier_map
