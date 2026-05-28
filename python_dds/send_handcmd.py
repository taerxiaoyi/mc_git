import time
from dataclasses import dataclass

import numpy as np
import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotations
import cyclonedds.idl.types as types

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.core import Qos, Policy


# =========================
# DDS config
# =========================
DOMAIN_ID = 1

TOPIC_NAME = "MocapUEHand"

# 发送频率（Hz）
PUBLISH_FPS = 30.0

# 是否循环播放
LOOP = True


# =========================
# DDS message definition
# =========================
@dataclass
@annotations.final
@annotations.autoid("sequential")
class MocapUEHandMsg(idl.IdlStruct, typename="MocapUEHandMsg"):
    handid: types.int32
    dof: types.int32
    timestamp: types.int64
    joints: types.array[types.float32, 60]
    enable: types.int32


class MocapUEHandMsgPublisher:
    def __init__(self):
        self.participant = DomainParticipant(domain_id=DOMAIN_ID)
        self.qos = Qos(
            Policy.History.KeepLast(depth=4),
        )
        self.topic = Topic(self.participant, TOPIC_NAME, MocapUEHandMsg, qos=self.qos)
        self.publisher = Publisher(self.participant)
        self.writer = DataWriter(self.publisher, self.topic)

    def publish(self, handid, dof, timestamp, joints, enable,):
        # joints = np.asarray(joints, dtype=np.float32).reshape(60)
        msg = MocapUEHandMsg(
            handid=handid,
            dof=dof,
            timestamp=timestamp,
            joints=joints,
            enable=enable,
        )
        self.writer.write(msg)

def main():
    publisher = MocapUEHandMsgPublisher()
    print(f"Publishing to topic: {TOPIC_NAME}, domain: {DOMAIN_ID}")

    frame_dt = 1.0 / PUBLISH_FPS

    while True:
        publisher.publish(
            handid=1,
            dof=14,
            timestamp=np.int64(time.time_ns()),
            joints=np.zeros(60, dtype=np.float32).tolist(),
            enable=1,
        )
        print(f"Published hand data at timestamp: {time.time_ns()}")
        time.sleep(frame_dt)

if __name__ == "__main__":
    main()