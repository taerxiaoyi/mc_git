# from __future__ import annotations
from dataclasses import dataclass
from collections import OrderedDict

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
TOPIC_NAME = "MocapUE5G115Topic1mpz"

@dataclass
@annotations.final
@annotations.autoid("sequential")
class MocapUE5G115Msg(idl.IdlStruct, typename="MocapUE5G115Msg"):
    fps: types.float32
    timestamp: types.int64
    xyz: types.array[types.float32, 15*3]
    wxyz: types.array[types.float32, 15*4]
    fingers: types.array[types.float32, 15] # [1] enable; [7] left hands; [7] right hands

class MocapUE5G115MsgSubscriber:
    def __init__(self):
        self.participant = DomainParticipant(domain_id=DOMAIN_ID)
        self.qos = Qos(
            Policy.History.KeepLast(depth=4),
        )
        print("types module path:", types.__file__)
        self.topic = Topic(self.participant, TOPIC_NAME, MocapUE5G115Msg, qos=self.qos)
        self.subscriber = Subscriber(self.participant)
        self.reader = DataReader(self.subscriber, self.topic)

    def subscribe(self):
        sample = self.reader.read()[-1]
        return sample
########################################################################################################################


def stickman_plot(ax, parents, stickman, color):
    x_vals = stickman[:, 0]
    y_vals = stickman[:, 1]
    z_vals = stickman[:, 2]
    ax.scatter(x_vals, y_vals, z_vals, c=color, marker='o')

    for i in range(len(parents)):
        p = parents[i]
        if p == -1: continue
        ax.plot([stickman[i, 0], stickman[p, 0]], 
                [stickman[i, 1], stickman[p, 1]], 
                [stickman[i, 2], stickman[p, 2]], color=color, linewidth=1)

    for i in range(len(parents)):
        xyz, quat = stickman[i, 0:3], stickman[i, 3:7][[1,2,3,0]] # to (x,y,z,w)
        rot_matrix = R.from_quat(quat).as_matrix()
        
        scale = 0.1
        ax.plot([xyz[0], xyz[0] + scale * rot_matrix[0,0]],
                [xyz[1], xyz[1] + scale * rot_matrix[1,0]], 
                [xyz[2], xyz[2] + scale * rot_matrix[2,0]], 'r-', linewidth=2)
        ax.plot([xyz[0], xyz[0] + scale * rot_matrix[0,1]],
                [xyz[1], xyz[1] + scale * rot_matrix[1,1]], 
                [xyz[2], xyz[2] + scale * rot_matrix[2,1]], 'g-', linewidth=2)
        ax.plot([xyz[0], xyz[0] + scale * rot_matrix[0,2]],
                [xyz[1], xyz[1] + scale * rot_matrix[1,2]], 
                [xyz[2], xyz[2] + scale * rot_matrix[2,2]], 'b-', linewidth=2)
        
    ax.set_xlim([x_vals[0]-1, x_vals[0]+1])
    ax.set_ylim([y_vals[0]-1, y_vals[0]+1])
    ax.set_zlim([0, 2])

import matplotlib
matplotlib.rcParams['toolbar'] = 'None'

if __name__ == "__main__":
    mocap = MocapUE5G115MsgSubscriber()
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.spatial.transform import Rotation as R
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    while True:
        msg = mocap.subscribe()
        print("fps:", msg.fps)
        print("timestamp:", msg.timestamp)
        
        parents = [-1, 0,1,2,3, 0,5,6,7, 0,9,10, 0,12,13]
        
        xyz = np.array(msg.xyz, dtype=np.float32).reshape(15,3)
        # xyz[:, :2] = xyz[:, :2] - xyz[:1, :2]
        wxyz = np.array(msg.wxyz, dtype=np.float32).reshape(15,4)
        stickman = np.concatenate([xyz, wxyz], axis=-1)

        # print(stickman)

        ax.clear()
        stickman_plot(ax, parents, stickman, "black")
        
        # plt.draw()
        plt.pause(0.001) 