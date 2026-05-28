# from __future__ import annotations
from dataclasses import dataclass
from collections import OrderedDict
from time import sleep

import numpy as np
# import threading
# import torch
# from pynput import keyboard

import os
import sys
sys.path.append(os.getcwd())

# from ei.eman import EI_ROOT_DIR
# from ei.eman.tasks.base_task import BaseTaskCfg, BaseTask
# from ei.eman.base.rotation_helper import get_heading_from_quat

########################################################################################################################
from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotations
import cyclonedds.idl.types as types

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
# from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.sub import Subscriber, DataReader
# from cyclonedds.util import duration
from cyclonedds.core import Qos, Policy

DOMAIN_ID = 1
TOPIC_NAME = "MocapUEHand"

@dataclass
@annotations.final
@annotations.autoid("sequential")
class MocapUEHandData_Msg(idl.IdlStruct, typename="MocapUEHandData_Msg"):
    handid: types.int32
    dof: types.int32
    timestamp: types.int64
    joints: types.array[types.float32, 60]
    enable: types.int32

class MocapUEHandSubscriber:
    def __init__(self):
        self.participant = DomainParticipant(domain_id=DOMAIN_ID)
        self.qos = Qos(
            Policy.History.KeepLast(depth=4),
        )
        print("types module path:", types.__file__)
        self.topic = Topic(self.participant, TOPIC_NAME, MocapUEHandData_Msg, qos=self.qos)
        self.subscriber = Subscriber(self.participant)
        self.reader = DataReader(self.subscriber, self.topic)

    def subscribe(self):
        sample = self.reader.read()[-1]
        return sample
########################################################################################################################

if __name__ == "__main__":
    mocap_ue_hand = MocapUEHandSubscriber()

    while True:
        sleep(0.1)
        msg = mocap_ue_hand.subscribe()
        print("timestamp:", msg.timestamp)
