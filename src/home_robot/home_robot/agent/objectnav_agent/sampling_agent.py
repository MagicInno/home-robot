# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.nn import DataParallel

import home_robot.utils.pose as pu

# from home_robot.core.abstract_agent import Agent
from home_robot.agent.objectnav_agent import ObjectNavAgent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.mapping.voxel import SparseVoxelMap
from home_robot.navigation_planner.rrt import RRTPlanner

from .objectnav_agent_module import ObjectNavAgentModule


class SamplingBasedObjectNavAgent(ObjectNavAgent):
    """
    SamplingBasedObjectNavAgent 类是 ObjectNavAgent 的一个子类，用于在虚拟环境中执行基于采样的目标导航任务。
    这个类主要利用 RRT (Rapidly-exploring Random Tree) 算法进行路径规划，并结合稀疏体素地图（SparseVoxelMap）来处理三维空间信息
    """
    """Simple object nav agent based on a 2D semantic map"""

    def __init__(self, config, device_id: int = 0):
        # 初始化 SamplingBasedObjectNavAgent 类的实例。除了从父类 ObjectNavAgent 继承的功能外，额外配置了基于 RRT 的规划器和稀疏体素地图。

        super(SamplingBasedObjectNavAgent, self).__init__(config, device_id)
        self.planner = RRTPlanner()
        self.voxel_map = SparseVoxelMap()

    def reset(self):
        # 重置代理的状态，清除体素地图中的信息，并重新初始化 RRT 规划器和其他组件。这用于每次开始新任务时清空之前的状态。

        """Clear information in the voxel map"""
        self.reset_vectorized()
        self.voxel_map.reset()
        self.planner.reset()
        self.episode_panorama_start_steps = self.panorama_start_steps

    def act(self, obs: Observations) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        # 根据当前观测数据决定代理的下一步行动。此方法目前尚未实现，其预期功能包括处理观测数据、更新地图、使用 RRT 规划器计算路径，并决定移动动作。

        """Use this action to move around in the world"""

        # 1 - Obs preprocessing
        (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            recep_goal_category,
            goal_name,
            camera_pose,
        ) = self._preprocess_obs(obs)

        raise NotImplementedError()

        return None, None
