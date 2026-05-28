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
# TOPIC_NAME = "MocapUE5G115Topicvla"

TOPIC_NAME = "MocapUE5G115Topic1mpz"

# 你的 npy 文件路径
NPY_PATH = "/media/mpz/d5f7a2a2-7dfb-4053-8e51-ee6943e25306/Downloads/pose7_seq.npy"

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
class MocapUE5G115Msg(idl.IdlStruct, typename="MocapUE5G115Msg"):
    fps: types.float32
    timestamp: types.int64
    xyz: types.array[types.float32, 15 * 3]
    wxyz: types.array[types.float32, 15 * 4]
    fingers: types.array[types.float32, 15]  # [1] enable; [7] left hands; [7] right hands


class MocapUE5G115MsgPublisher:
    def __init__(self):
        self.participant = DomainParticipant(domain_id=DOMAIN_ID)
        self.qos = Qos(
            Policy.History.KeepLast(depth=4),
        )
        self.topic = Topic(self.participant, TOPIC_NAME, MocapUE5G115Msg, qos=self.qos)
        self.publisher = Publisher(self.participant)
        self.writer = DataWriter(self.publisher, self.topic)

    def publish(self, fps, xyz_15x3, wxyz_15x4, fingers_15=None):
        xyz_15x3 = np.asarray(xyz_15x3, dtype=np.float32).reshape(15, 3)
        wxyz_15x4 = np.asarray(wxyz_15x4, dtype=np.float32).reshape(15, 4)

        if fingers_15 is None:
            fingers_15 = np.zeros(15, dtype=np.float32)
        else:
            fingers_15 = np.asarray(fingers_15, dtype=np.float32).reshape(15)

        msg = MocapUE5G115Msg(
            fps=np.float32(fps),
            timestamp=np.int64(time.time_ns()),
            xyz=xyz_15x3.reshape(-1).tolist(),
            wxyz=wxyz_15x4.reshape(-1).tolist(),
            fingers=fingers_15.tolist(),
        )
        self.writer.write(msg)


# =========================
# Pose7 selection logic
# pose7 row format:
# [qw, qx, qy, qz, x, y, z]
# =========================
SELECTED_INDICES = [
    0,   # root

    3,   # left_hip_yaw
    4,   # left_knee
    5,   # left_ankle_pitch
    6,   # left_ankle_roll

    9,   # right_hip_yaw
    10,  # right_knee
    11,  # right_ankle_pitch
    12,  # right_ankle_roll

    18,  # left_shoulder_yaw
    19,  # left_elbow
    22,  # left_wrist_yaw

    25,  # right_shoulder_yaw
    26,  # right_elbow
    29,  # right_wrist_yaw
]


def load_pose7_seq(npy_path):
    arr = np.load(npy_path)

    if arr.ndim == 2:
        # single frame: (30, 7) -> (1, 30, 7)
        if arr.shape != (30, 7):
            raise ValueError(f"Expected single-frame shape (30, 7), got {arr.shape}")
        arr = arr[None, ...]

    if arr.ndim != 3 or arr.shape[1:] != (30, 7):
        raise ValueError(f"Expected shape (T, 30, 7), got {arr.shape}")

    return arr.astype(np.float32)


def pick_15_points_from_pose7(frame_30x7):
    """
    Input:
        frame_30x7: shape (30, 7)
        row format: [qw, qx, qy, qz, x, y, z]

    Output:
        xyz_15x3, wxyz_15x4
    """
    selected = frame_30x7[SELECTED_INDICES]   # (15, 7)

    wxyz = selected[:, 0:4]   # (15, 4)
    xyz = selected[:, 4:7]    # (15, 3)

    return xyz, wxyz


def main():
    pose7_seq = load_pose7_seq(NPY_PATH)
    publisher = MocapUE5G115MsgPublisher()

    print(f"Loaded pose7_seq: shape = {pose7_seq.shape}")
    print(f"Publishing to topic: {TOPIC_NAME}, domain: {DOMAIN_ID}")

    frame_dt = 1.0 / PUBLISH_FPS

    try:
        while True:
            for i in range(pose7_seq.shape[0]):
                frame = pose7_seq[i]  # (30, 7)

                xyz_15x3, wxyz_15x4 = pick_15_points_from_pose7(frame)

                # 这里先全部置 0，后面如果你要传手指开关再改
                fingers_15 = np.zeros(15, dtype=np.float32)

                publisher.publish(
                    fps=PUBLISH_FPS,
                    xyz_15x3=xyz_15x3,
                    wxyz_15x4=wxyz_15x4,
                    fingers_15=fingers_15,
                )

                print(f"[send] frame={i:04d}/{pose7_seq.shape[0]}")

                time.sleep(frame_dt)

            if not LOOP:
                break

    except KeyboardInterrupt:
        print("Stopped by user.")


if __name__ == "__main__":
    main()