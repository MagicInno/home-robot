# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
查看关键帧图像（view_keyframe_imgs 函数）:

这个函数从一个 HDF5 文件中读取指定试验的关键帧图像，并使用 Matplotlib 显示这些图像。
file_object 是一个打开的 HDF5 文件对象，trial_name 是试验的名称。
代码循环遍历每个关键帧，读取图像数据并将其转换为可视化的图像格式，然后使用 plt.imshow(img) 显示图像。
在 RVIZ 中绘制执行器姿态（plot_ee_pose 函数）:

这个函数从 HDF5 文件中提取每个关键帧的执行器（例如机器人手臂末端）的位置和旋转，然后将这些姿态作为 TF（变换）消息发送到 ROS 中，以便在 RVIZ 中可视化。
它同样接受一个 HDF5 文件对象和试验名称，还需要一个 ROS TF 广播器（ros_pub）。
对于每个关键帧，函数读取执行器的位置和旋转，然后创建一个 TransformStamped 消息，其中包含这些信息。
这个消息被发送到 ROS，可在 RVIZ 中查看。函数在每次发送消息后等待用户输入，以逐一查看每个关键帧的姿态。
"""

from typing import List, Tuple

import h5py
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped
from matplotlib import pyplot as plt
from tf2_ros import tf2_ros

from home_robot.utils.data_tools.image import img_from_bytes


def view_keyframe_imgs(file_object: h5py.File, trial_name: str):
    """utility to view keyframe images for named trial from h5 file"""
    num_keyframes = len(file_object[f"{trial_name}/head_rgb"].keys())
    for i in range(num_keyframes):
        _key = f"{trial_name}/head_rgb/{i}"
        img = img_from_bytes(file_object[_key][()])
        plt.imshow(img)
        plt.show()


def plot_ee_pose(
    file_object: h5py.File, trial_name: str, ros_pub: tf2_ros.TransformBroadcaster
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """plot keyframes as TF to be visualized in RVIZ, also return the ee pose associated with them"""
    num_keyframes = len(file_object[f"{trial_name}/ee_pose"][()])
    ee_pose = []
    for i in range(num_keyframes):
        pos = file_object[f"{trial_name}/ee_pose"][()][i][:3]
        rot = file_object[f"{trial_name}/ee_pose"][()][i][3:]
        ee_pose.append((pos, rot))
        pose_message = TransformStamped()
        pose_message.header.stamp = rospy.Time.now()
        pose_message.header.frame_id = "base_link"

        pose_message.child_frame_id = f"key_frame_{i}"
        pose_message.transform.translation.x = pos[0]
        pose_message.transform.translation.y = pos[1]
        pose_message.transform.translation.z = pos[2]

        pose_message.transform.rotation.x = rot[0]
        pose_message.transform.rotation.y = rot[1]
        pose_message.transform.rotation.z = rot[2]
        pose_message.transform.rotation.w = rot[3]

        ros_pub.sendTransform(pose_message)
        input("Press enter to continue")

    return ee_pose
