# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import gym.spaces as spaces
import numpy as np

"""
获取机器人或仿真环境中所需的所有观测空间（observation space）的超集。观测空间是一个关键概念，
它定义了机器人感知其环境的所有可能方式。在强化学习和机器人控制中，观测空间通常包括机器人通过其传感器能够观测到的所有数据。


观测类型:

is_holding：表示机器人是否抓住了某个物体。
head_depth：代表机器人头部的深度感知。
joint：机器人的关节状态。
object_embedding：物体的嵌入表示，用于物体识别或分类。
relative_resting_position：相对休息位置。
object_segmentation、goal_recep_segmentation、receptacle_segmentation、start_recep_segmentation：用于不同目标的语义分割。
ovmm_nav_goal_segmentation：用于导航目标的语义分割。
robot_start_gps 和 robot_start_compass：机器人的初始位置和方向。
start_receptacle 和 goal_receptacle：起始和目标容器。
空间类型:

使用 gym.spaces.Box 定义具有最小值、最大值、形状和数据类型的连续空间。
这些空间可以代表真实值（如深度信息）、分类标签或二进制标志（如是否抓住物体）。
应用:

这个函数可以在机器人硬件或仿真环境中使用，为机器人或智能体提供全面的感知信息。
通过这些观测空间，机器人可以更好地理解其环境，并据此做出决策。

"""
def get_complete_obs_space(skill_config, baseline_config):
    """
    Get superset of observation space needed for any policy.
    This avoids needing to use the habitat configs to import the observation space on hardware.
    TODO: Find way to import observation space from regular YAML configs to avoid this hardcoding.
    """
    return spaces.dict.Dict(
        {
            "is_holding": spaces.Box(0.0, 1.0, (1,), np.float32),
            "head_depth": spaces.Box(
                0.0,
                1.0,
                (skill_config.sensor_height, skill_config.sensor_width, 1),
                np.float32,
            ),
            "joint": spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (10,),
                np.float32,
            ),
            "object_embedding": spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (512,),
                np.float32,
            ),
            "relative_resting_position": spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (3,),
                np.float32,
            ),
            "object_segmentation": spaces.Box(
                0.0,
                1.0,
                (skill_config.sensor_height, skill_config.sensor_width, 1),
                np.uint8,
            ),
            "goal_recep_segmentation": spaces.Box(
                0.0,
                1.0,
                (skill_config.sensor_height, skill_config.sensor_width, 1),
                np.uint8,
            ),
            "ovmm_nav_goal_segmentation": spaces.Box(
                0.0,
                1.0,
                (
                    skill_config.sensor_height,
                    skill_config.sensor_width,
                    skill_config.nav_goal_seg_channels,
                ),
                np.int32,
            ),
            "receptacle_segmentation": spaces.Box(
                0.0,
                1.0,
                (skill_config.sensor_height, skill_config.sensor_width, 1),
                np.uint8,
            ),
            "robot_start_gps": spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (2,),
                np.float32,
            ),
            "robot_start_compass": spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (1,),
                np.float32,
            ),
            "start_recep_segmentation": spaces.Box(
                0.0,
                1.0,
                (skill_config.sensor_height, skill_config.sensor_width, 1),
                np.uint8,
            ),
            "start_receptacle": spaces.Box(
                0,
                baseline_config.ENVIRONMENT.num_receptacles - 1,
                (1,),
                np.int64,
            ),
            "goal_receptacle": spaces.Box(
                0,
                baseline_config.ENVIRONMENT.num_receptacles - 1,
                (1,),
                np.int64,
            ),
        }
    )
