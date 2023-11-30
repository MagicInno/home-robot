# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple

import numpy as np

from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import (
    ContinuousFullBodyAction,
    DiscreteNavigationAction,
    Observations,
)

"""
这段代码定义了一个名为 `RandomAgent` 的智能体类，它从 `home_robot.core.abstract_agent.Agent` 类继承。
`RandomAgent` 是一个随机智能体，它会在给定的动作空间内随机选择动作。以下是其主要特点：

1. **初始化**: 构造函数初始化智能体，并设置一些关键参数，例如抓取和释放物体的概率、停止的概率、最大前进距离、最大转向角度等。

2. **动作选择**: `act` 函数是智能体的主要函数，它根据当前的观测数据选择一个动作。动作可以是离散的（例如移动前进、左转、右转等）
或连续的（包括机械臂关节的移动和基本的XYT移动）。

3. **随机动作**: 智能体会随机决定是进行抓取、释放物体、停止或其他动作。这是通过生成一个随机数并根据预定义的概率阈值来决定的。

4. **复位功能**: `reset`, `reset_vectorized`, 和 `reset_vectorized_for_env` 函数用于在任务开始或特定环境下重置智能体的内部状态。

`RandomAgent` 类适用于测试和基准场景，其中智能体的行为不是由复杂的策略或模型驱动，而是完全随机的。
这种类型的智能体可以用来评估环境的难度或作为其他更复杂智能体的对照组。
"""
class RandomAgent(Agent):
    """A random agent that takes random discrete or continuous actions."""

    def __init__(self, config, device_id: int = 0):
        super().__init__()
        self.config = config
        self.snap_probability = 5e-3
        self.desnap_probability = 5e-3
        self.stop_probability = 0.01
        self.max_forward = (
            config.habitat.task.actions.base_velocity.max_displacement_along_axis
        )
        self.max_turn_degrees = (
            config.habitat.task.actions.base_velocity.max_turn_degrees
        )
        self.max_turn_radians = self.max_turn_degrees / 180 * np.pi
        self.max_joints_delta = config.habitat.task.actions.arm_action.max_delta_pos
        self.discrete_actions = config.AGENT.PLANNER.discrete_actions
        self.timestep = 0
        assert (
            self.snap_probability + self.desnap_probability + self.stop_probability
            <= 1.0
        )

    def reset(self):
        """Initialize agent state."""
        self.timestep = 0

    def reset_vectorized(self):
        """Initialize agent state."""
        self.timestep = 0

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.timestep = 0

    def act(
        self, obs: Observations
    ) -> Tuple[DiscreteNavigationAction, Dict[str, Any], Observations]:
        """Take a random action."""
        action = None
        r = np.random.rand()
        info = {"timestep": self.timestep, "semantic_frame": obs.rgb}
        if r < self.snap_probability:
            action = DiscreteNavigationAction.SNAP_OBJECT
        elif r < self.snap_probability + self.desnap_probability:
            action = DiscreteNavigationAction.DESNAP_OBJECT
        elif (
            r < self.snap_probability + self.desnap_probability + self.stop_probability
        ):
            action = DiscreteNavigationAction.STOP
        elif self.discrete_actions:
            action = np.random.choice(
                [
                    DiscreteNavigationAction.MOVE_FORWARD,
                    DiscreteNavigationAction.TURN_LEFT,
                    DiscreteNavigationAction.TURN_RIGHT,
                    DiscreteNavigationAction.EXTEND_ARM,
                    DiscreteNavigationAction.NAVIGATION_MODE,
                    DiscreteNavigationAction.MANIPULATION_MODE,
                ]
            )
        else:
            xyt = np.random.uniform(
                [-self.max_forward, -self.max_forward, -self.max_turn_radians],
                [self.max_forward, self.max_forward, self.max_turn_radians],
            )
            joints = np.random.uniform(
                -self.max_joints_delta, self.max_joints_delta, size=(10,)
            )
            action = ContinuousFullBodyAction(joints, xyt)
        self.timestep += 1
        return action, info, obs
