import time
import threading
import numpy as np
import mujoco
import mujoco.viewer
import os

# 导入 SDK 相关模块
from westlake_sdkpy.core.channel import ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
from westlake_sdkpy.idl.agile.msg.dds_ import LowCmd_, LowState_
from westlake_sdkpy.idl.default import agile_msg_dds__LowState_

# 话题定义
TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"

class MujocoSimNode:
    def __init__(self, xml_path):
        # 路径检查
        if not os.path.exists(xml_path):
            current_dir = os.getcwd()
            full_path = os.path.join(current_dir, xml_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"找不到模型文件: {xml_path}")
            xml_path = full_path

        # 1. 初始化 Mujoco
        print(f"[Mujoco] Loading model (Kinematic Mode) from: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.nu = self.model.nu
        print(f"[Mujoco] Model loaded. Actuators: {self.nu}")

        # 2. 指令缓存
        self.latest_cmd = None
        self.cmd_lock = threading.Lock()
        
        # 3. 初始化 DDS
        ChannelFactoryInitialize(0)
        
        self.sub = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.sub.Init(handler=self.on_low_cmd_received, queueLen=10)
        
        self.pub_state = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.pub_state.Init()
        
        self.low_state_msg = agile_msg_dds__LowState_()
        self.sequence_counter = 0

        print(f"[DDS] Listening on {TOPIC_LOWCMD}")
        
        self.running = True

    def on_low_cmd_received(self, msg: LowCmd_):
        with self.cmd_lock:
            self.latest_cmd = msg

    def publish_sim_state(self):
        """发送状态反馈"""
        # self.sequence_counter += 1
        self.sequence_counter = (self.sequence_counter + 1) & 0xFFFF   # wrap around 0 ~ 65535
        self.low_state_msg.sequences = self.sequence_counter
        
        # 1. 填充关节状态
        for i in range(32):
            motor = self.low_state_msg.motor_state[i]
            if i < self.nu:
                joint_id = self.model.actuator(i).trnid[0]
                qpos_idx = self.model.jnt_qposadr[joint_id]
                qvel_idx = self.model.jnt_dofadr[joint_id]
                
                motor.q = self.data.qpos[qpos_idx]
                motor.dq = self.data.qvel[qvel_idx]
                motor.tau_est = 0.0 
            else:
                motor.q = 0.0; motor.dq = 0.0; motor.tau_est = 0.0

        # 2. 填充 IMU
        if self.model.nq >= 7:
            self.low_state_msg.imu_state.quaternion = self.data.qpos[3:7]
            self.low_state_msg.imu_state.gyroscope = self.data.qvel[3:6]
            self.low_state_msg.imu_state.accelerometer = self.data.qacc[0:3]
        
        self.pub_state.Write(self.low_state_msg)

        # 打印调试信息 (每 50 帧一次，约 1 秒)
        if self.sequence_counter % 50 == 0:
            self.print_cmd_table()

    def print_cmd_table(self):
        """【核心修改】以表格形式打印每个关节的详细指令"""
        with self.cmd_lock:
            cmd = self.latest_cmd
            
        if cmd is None:
            print(f"\n[Seq {self.sequence_counter}] Waiting for CMD...")
            return

        print(f"\n[Seq {self.sequence_counter}] ==== Received LowCmd Detail ====")
        # 打印表头
        # ID | TargetPos | TargetVel | Kp | Kd | Tau
        header = f"{'ID':<3} | {'Targ Q (rad)':<12} | {'Targ DQ':<10} | {'Kp':<5} | {'Kd':<5} | {'Tau':<6}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        # 遍历所有有效电机
        for i in range(self.nu):
            m = cmd.motor_cmd[i]
            # 格式化打印每一行
            print(f"{i:<3} | {m.q:<12.4f} | {m.dq:<10.3f} | {m.kp:<5.1f} | {m.kd:<5.1f} | {m.tau:<6.3f}")
        
        print("=" * len(header))

    def apply_kinematics(self):
        """运动学更新 (无惯性模式)"""
        with self.cmd_lock:
            cmd = self.latest_cmd

        if cmd is None:
            return

        for i in range(self.nu):
            if i >= 32: break

            motor_cmd = cmd.motor_cmd[i]
            
            # 找到对应的关节地址
            joint_id = self.model.actuator(i).trnid[0]
            qpos_idx = self.model.jnt_qposadr[joint_id]
            qvel_idx = self.model.jnt_dofadr[joint_id]

            # 强制瞬移
            self.data.qpos[qpos_idx] = motor_cmd.q
            self.data.qvel[qvel_idx] = motor_cmd.dq
            self.data.ctrl[i] = 0.0

    def run(self):
        print(f"[Sim] Kinematic Loop started. Waiting for deploy_real.py...")
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                self.apply_kinematics()
                mujoco.mj_forward(self.model, self.data) # 使用 forward 仅计算几何
                
                # 手动推进时间
                self.data.time += self.model.opt.timestep

                self.publish_sim_state()
                viewer.sync()

                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == "__main__":
    XML_PATH = "resources/robots/o1/o1_fixed.xml"
    try:
        node = MujocoSimNode(XML_PATH)
        node.run()
    except Exception as e:
        print(f"Error: {e}")