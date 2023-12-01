# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import dataclasses
import os
import queue

import a0


@dataclasses.dataclass(order=True)
class ReplayMessage:
    """
    使用dataclasses定义了一个名为ReplayMessage的数据类，用于存储重放消息的相关信息。
包含的字段有：
timestamp：消息的时间戳。
pkt（a0.Packet类型）：消息数据包。
path：消息来源的路径。
topic：消息的主题。
reader（a0.ReaderSync类型）：用于读取消息的同步读取器。
    """
    timestamp: int
    pkt: a0.Packet = dataclasses.field(compare=False)
    path: str = dataclasses.field(compare=False)
    topic: str = dataclasses.field(compare=False)
    reader: a0.ReaderSync = dataclasses.field(compare=False)


class ReplayManager:
    """
    __init__方法：初始化ReplayManager实例。
参数read_paths是一个路径列表，每个路径指向一个消息记录文件。
创建一个名为_srcs的字典，其中每个读取路径都映射到一个a0.ReaderSync实例，用于同步读取那个路径下的消息。
使用queue.PriorityQueue创建一个优先级队列_pq，用于存储并排序即将重放的消息。
    """
    def __init__(self, read_paths):
        self._srcs = {
            read_path: a0.ReaderSync(a0.File(read_path), a0.INIT_OLDEST)
            for read_path in read_paths
        }

        self._pq = queue.PriorityQueue()
        for path, reader in self._srcs.items():
            self._load_next_message(path, reader)

    def _load_next_message(self, path, reader):
        if not reader.can_read():
            return

        pkt = reader.read()
        timestamp = int(dict(pkt.headers)["a0_time_mono"])
        topic = os.path.basename(path).split(".")[0]
        self._pq.put(ReplayMessage(timestamp, pkt, path, topic, reader))

    def can_read(self):
        return not self._pq.empty()

    def read(self):
        replay_message = self._pq.get()
        self._load_next_message(replay_message.path, replay_message.reader)
        return replay_message

    def __iter__(self):
        return self

    def __next__(self):
        if not self.can_read():
            raise StopIteration
        return self.read()
