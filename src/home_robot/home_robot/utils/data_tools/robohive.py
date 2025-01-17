# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


""" =================================================
original Copyright (C) 2018 Vikash Kumar
- updated by Chris Paxton
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import os
import pickle
import time

import click
import gym
import numpy as np

# Import the writer to save out data to hdf5
from data_tools.writer import DataWriter
from mj_envs.utils.viz_paths import plot_paths as plotnsave_paths
"""
检查和分析环境（例如机器学习模型中的模拟环境）及其相关策略（策略可以是一个预先训练好的模型或者是一个随机策略）。工具的主要功能包括在屏幕上或离屏渲染环境，
保存模拟路径为Pickle格式或者为2D图像，以及其他一些可视化和数据记录功能。主要步骤和功能如下：

命令行选项:

env_name: 要加载的环境名称。
policy_path: 策略文件的路径。
mode: 策略的模式（探索或评估）。
seed: 生成环境实例的随机种子。
num_episodes: 要可视化的剧集数。
render: 在屏幕上或离屏渲染环境。
camera_name: 渲染时使用的相机名称。
output_dir: 保存输出的目录。
output_name: 保存输出的名称。
save_paths: 是否保存模拟路径。
plot_paths: 是否绘制路径的2D图像。
初始化和加载环境:

根据提供的种子初始化随机生成器。
加载指定的环境。
解析策略和输出:

如果提供了策略路径，则从该路径加载策略。
如果未提供策略路径，则使用随机策略。
模拟和数据记录:

使用策略在环境中进行模拟。
记录每一步的观察结果、动作、奖励和完成状态。
可选择将模拟路径保存为Pickle文件或绘制为2D图像。
额外的可视化和保存选项:

根据命令行选项在屏幕上或离屏渲染环境。
保存模拟路径到指定的目录
"""
DESC = """
Helper script to examine an environment and associated policy for behaviors; \n
- either onscreen, or offscreen, or just rollout without rendering.\n
- save resulting paths as pickle or as 2D plots
USAGE:\n
    $ python examine_env.py --env_name door-v0 \n
    $ python examine_env.py --env_name door-v0 --policy my_policy.pickle --mode evaluation --episodes 10 \n
"""


# Random policy
class rand_policy:
    def __init__(self, env, seed):
        self.env = env
        self.env.action_space.np_random.seed(seed)  # requires exlicit seeding

    def get_action(self, obs):
        # return self.env.np_random.uniform(high=self.env.action_space.high, low=self.env.action_space.low)
        return self.env.action_space.sample(), {"mode": "random samples"}


# MAIN =========================================================
@click.command(help=DESC)
@click.option("-e", "--env_name", type=str, help="environment to load", required=True)
@click.option(
    "-p",
    "--policy_path",
    type=str,
    help="absolute path of the policy file",
    default=None,
)
@click.option(
    "-m",
    "--mode",
    type=str,
    help="exploration or evaluation mode for policy",
    default="evaluation",
)
@click.option(
    "-s",
    "--seed",
    type=int,
    help="seed for generating environment instances",
    default=123,
)
@click.option(
    "-n", "--num_episodes", type=int, help="number of episodes to visualize", default=10
)
@click.option(
    "-r",
    "--render",
    type=click.Choice(["onscreen", "offscreen", "none"]),
    help="visualize onscreen or offscreen",
    default="onscreen",
)
@click.option(
    "-c", "--camera_name", type=str, default=None, help=("Camera name for rendering")
)
@click.option(
    "-o", "--output_dir", type=str, default="./", help=("Directory to save the outputs")
)
@click.option(
    "-on",
    "--output_name",
    type=str,
    default=None,
    help=("The name to save the outputs as"),
)
@click.option(
    "-sp", "--save_paths", type=bool, default=False, help=("Save the rollout paths")
)
@click.option(
    "-pp",
    "--plot_paths",
    type=bool,
    default=False,
    help=("2D-plot of individual paths"),
)
@click.option(
    "-ea",
    "--env_args",
    type=str,
    default=None,
    help=("env args. E.g. --env_args \"{'is_hardware':True}\""),
)
def main(
    env_name,
    policy_path,
    mode,
    seed,
    num_episodes,
    render,
    camera_name,
    output_dir,
    output_name,
    save_paths,
    plot_paths,
    env_args,
):

    # seed and load environments
    np.random.seed(seed)
    env = (
        gym.make(env_name)
        if env_args is None
        else gym.make(env_name, **(eval(env_args)))
    )
    env.seed(seed)

    # resolve policy and outputs
    if policy_path is not None:
        pi = pickle.load(open(policy_path, "rb"))
        if output_dir == "./":  # overide the default
            output_dir, pol_name = os.path.split(policy_path)
            if output_name is None:
                output_name = os.path.splitext(pol_name)[0]
    else:
        pi = rand_policy(env, seed)
        mode = "exploration"
        output_name = "random_policy"

    # resolve directory
    if (not os.path.isdir(output_dir)) and (
        render == "offscreen" or save_paths or plot_paths is not None
    ):
        os.mkdir(output_dir)

    # save paths
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(output_dir, output_name + "{}.h5".format(time_stamp))
    writer = DataWriter(filename)
    for i in range(num_episodes):
        # examine policy's behavior to recover paths

        done = False
        horizon = env.spec.max_episode_steps
        o = env.reset()
        t = 0
        while t < horizon and done is False:
            a = (
                pi.get_action(o)[0]
                if mode == "exploration"
                else pi.get_action(o)[1]["evaluation"]
            )
            next_o, rwd, done, env_info = env.step(a)
            writer.add_frame(observation=o, action=a, reward=rwd, done=done)
            o = next_o
            t += 1
        # Write the final observation vector
        writer.add_frame(observation=o)
        writer.write_trial()
        # paths = env.examine_policy(
        #    policy=pi,
        #    horizon=env.spec.max_episode_steps,
        #    num_episodes=1, # num_episodes,
        #    frame_size=(640,480),
        #    mode=mode,
        #    output_dir=output_dir+'/',
        #    filename=output_name,
        #    camera_name=camera_name,
        #    render=render)
        # if save_paths:
        #    #file_name = output_dir + '/' + output_name + '{}_paths.pickle'.format(time_stamp)
        #    #pickle.dump(paths, open(file_name, 'wb'))
        #    #print("saved ", file_name)
        #    #import pdb; pdb.set_trace()

    # plot paths
    if plot_paths:
        file_name = output_dir + "/" + output_name + "{}".format(time_stamp)
        plotnsave_paths(plot_paths, env=env, fileName_prefix=file_name)


if __name__ == "__main__":
    main()
