# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import threading
from typing import Callable, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from home_robot.utils.config import get_control_config
from home_robot.utils.geometry import xyt_global_to_base

DEFAULT_CFG_NAME = "traj_follower"

"""
这段代码定义了一个轨迹跟随器（TrajFollower），它是用于导航机器人的控制器。
该控制器接受一个轨迹函数作为输入，该函数计算在给定时间点上机器人的期望状态（位置和速度）。
然后，控制器使用反馈控制算法计算机器人的速度命令（线速度和角速度）以跟踪这个轨迹。
"""
class TrajFollower:
    def __init__(self, cfg: Optional["DictConfig"] = None):
        # 加载配置，设置比例增益（kp）和积分增益（ki）。初始化用于更新轨迹的锁和轨迹缓冲区。

        if cfg is None:
            cfg = get_control_config(DEFAULT_CFG_NAME)
        else:
            self.cfg = cfg

        # Compute gain
        self.kp = cfg.k_p
        self.ki = (cfg.damp_ratio * self.kp) ** 2 / 4.0

        # Init
        self._traj_update_lock = threading.Lock()

        self._is_done = True
        self.traj = None
        self.traj_buffer = None

        self.e_int = np.zeros(3)
        self._t_prev = 0

    def update_trajectory(
        self, traj: Callable[[float], Tuple[np.ndarray, np.ndarray, bool]]
    ):
        """
        更新轨迹 (update_trajectory 方法):

        允许外部函数或进程更新轨迹函数。
        使用锁来确保线程安全。
        """
        with self._traj_update_lock:
            self.traj_buffer = traj

    def is_done(self) -> bool:
        """检查是否完成 (is_done 方法):

检查控制器是否已完成轨迹跟踪。"""
        return self._is_done

    def forward(self, xyt: np.ndarray, t: float) -> Tuple[float, float]:
        """Returns velocity control command (v, w)
        计算控制命令 (forward 方法):

查询当前时间点的轨迹以获取期望状态（位置和速度）。
如果轨迹完成，则返回零速度。
使用反馈控制算法计算速度命令。
        """
        # Check for trajectory updates
        if self.traj_buffer is not None:
            with self._traj_update_lock:
                self.traj = self.traj_buffer
                self.traj_buffer = None
            self._is_done = False

        # Return zero velocites if no trajectory is active
        if self._is_done:
            return 0.0, 0.0

        # Query trajectory for desired states
        xyt_traj, dxyt_traj, done = self.traj(t)
        if done:
            self._is_done = True

        # Feedback control
        dt = t - self._t_prev
        self._t_prev = t
        v, w = self._feedback_controller(xyt_traj, dxyt_traj, xyt, dt)

        return v, w

    def _feedback_controller(
        self, xyt_des: np.ndarray, dxyt_des: np.ndarray, xyt_curr: np.ndarray, dt: float
    ) -> Tuple[float, float]:
        """
        反馈控制器 (_feedback_controller 方法):

计算期望状态和当前状态之间的误差。
使用比例-积分（PI）控制法则来计算误差导数的期望值。
计算速度命令以实现期望的误差导数。

        """
        # Compute reference input
        u_ref = np.array([np.linalg.norm(dxyt_des[:2]), dxyt_des[2]])

        # Compute error in local frame
        e = xyt_global_to_base(xyt_des, xyt_curr)

        # Compute desired error derivative via PI control
        self.e_int = self.cfg.decay * self.e_int + e * dt
        de_des = -self.kp * e - self.ki * self.e_int

        # Compute velocity feedback commands to achieve desired error derivative
        M_u2e = np.array([[-1, e[1]], [0, -e[0]], [0, -1]])
        M_ur2e = np.array([[np.cos(e[2]), 0], [np.sin(e[2]), 0], [0, 1]])
        u_output = np.linalg.pinv(M_u2e) @ (de_des - M_ur2e @ u_ref)

        return u_output[0], u_output[1]
