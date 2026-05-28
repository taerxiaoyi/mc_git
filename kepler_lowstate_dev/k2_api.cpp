#include <chrono>
#include <thread>
#include "utility/logger.h"
#include "utility/data_buffer.h"
#include "utility/real/unitree_tools.h"
#include "sim2/real/robot_backend/robot_backend_registry.h"
// #include "tasks/utils/mocap/MocapUEMsg.h"
// #include "custom/robot/channel/channel_publisher.hpp"
// namespace custom = alan::robot::channel;

// DDS
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// IDL
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

// std::shared_ptr<BaseRobotConfig> cfg = nullptr;

int jointDim_ = 28, handsDim_ = 0;
DataBuffer<CustomTypes::RobotBackendRawState> raw_state_buffer_; // interface with sim2real env
DataBuffer<CustomTypes::RobotBackendAction> raw_action_buffer_;  // interface with simulate robot

uint8_t mode_machine_;
Mode mode_pr_;
LowCmd_ low_cmd_;
LowState_ low_state_;
std::unique_ptr<ChannelPublisher<LowCmd_>> lowcmd_publisher_;               // publisher
std::unique_ptr<ChannelSubscriber<LowState_>> lowstate_subscriber_;        // subscriber
std::shared_ptr<unitree::robot::b2::MotionSwitcherClient> msc_;            // shut down motion related service

// // hands related
// HandCmd_ leftHand_cmd_;
// HandState_ leftHand_state_;
// std::unique_ptr<ChannelPublisher<HandCmd_>> leftHandcmd_publisher_;             // publisher
// std::unique_ptr<ChannelSubscriber<HandState_>> leftHandstate_subscriber_;         // subscriber

// HandCmd_ rightHand_cmd_;
// HandState_ rightHand_state_;
// std::unique_ptr<ChannelPublisher<HandCmd_>> rightHandcmd_publisher_;              // publisher
// std::unique_ptr<ChannelSubscriber<HandState_>> rightHandstate_subscriber_;          // subscriber

// for simulate robot
std::unique_ptr<ChannelSubscriber<LowCmd_>> lowcmd_subscriber_;             
std::unique_ptr<ChannelPublisher<LowState_>> lowstate_publisher_; 

void LowCmdHgHandler(const void *message) {
    const LowCmd_& msg = *(const LowCmd_ *)message;

    std::cout << "11111" << std::endl;

    // CRC verify
    if (msg.crc() != unitree_tools::Crc32Core((uint32_t *)&msg, (sizeof(msg) >> 2) - 1)) {
    FRC_ERROR("[UnitreeG1Backend.LowCmdHgHandler] CRC Error");
    std::exit(1);
    return;
    }

    CustomTypes::RobotBackendAction raw_action;
    raw_action = CustomTypes::RobotBackendAction(jointDim_, handsDim_);
    for (int i = 0; i < jointDim_; ++i) {
    raw_action.joint_pos[i] = msg.motor_cmd()[i].q();
    // std::cout << msg.motor_cmd()[i].q() << std::endl;
    raw_action.joint_vel[i] = msg.motor_cmd()[i].dq();
    raw_action.kp[i]        = msg.motor_cmd()[i].kp();
    raw_action.kd[i]        = msg.motor_cmd()[i].kd();
    raw_action.torque[i]    = msg.motor_cmd()[i].tau();
    }

    std::cout << "===== RobotBackendAction =====" << std::endl;

    std::cout << "joint_pos  (" << raw_action.joint_pos.size() << "): "
                << raw_action.joint_pos.transpose() << std::endl;

    std::cout << "joint_vel  (" << raw_action.joint_vel.size() << "): "
                << raw_action.joint_vel.transpose() << std::endl;

    std::cout << "kp         (" << raw_action.kp.size() << "): "
                << raw_action.kp.transpose() << std::endl;

    std::cout << "kd         (" << raw_action.kd.size() << "): "
                << raw_action.kd.transpose() << std::endl;

    std::cout << "torque     (" << raw_action.torque.size() << "): "
                << raw_action.torque.transpose() << std::endl;

    std::cout << "================================" << std::endl;

    raw_action_buffer_.SetData(raw_action);

}

std::thread publish_thread_;
std::atomic<bool> publish_running_{false};
// =============================
// 发送线程函数
// =============================
void PublishLoop()
{
    LowState_ dds_low_state;

    auto next_time = std::chrono::steady_clock::now();

    while (publish_running_.load())
    {
        next_time += std::chrono::milliseconds(10);

        dds_low_state.imu_state().quaternion()[0] = 1;                
        dds_low_state.imu_state().quaternion()[1] = 0;                
        dds_low_state.imu_state().quaternion()[2] = 0;                
        dds_low_state.imu_state().quaternion()[3] = 0;
        
        dds_low_state.imu_state().gyroscope()[0] = 1;
        dds_low_state.imu_state().gyroscope()[1] = 1;
        dds_low_state.imu_state().gyroscope()[2] = 1;
                        
        for (int i = 0; i < jointDim_; ++i) {
            dds_low_state.motor_state().at(i).q() = 1;
            dds_low_state.motor_state().at(i).dq() = 1;
        }

        dds_low_state.crc() = unitree_tools::Crc32Core((uint32_t *)&dds_low_state, (sizeof(dds_low_state) >> 2) - 1);

        lowstate_publisher_->Write(dds_low_state);

        std::this_thread::sleep_until(next_time);
    }
}
// =============================
// 启动线程
// =============================
void StartPublishThread()
{
    publish_running_ = true;
    publish_thread_ = std::thread(PublishLoop);
}
// =============================
// 停止线程
// =============================
void StopPublishThread()
{
    publish_running_ = false;
    if (publish_thread_.joinable())
        publish_thread_.join();
}



int main() {

    // g1 and h1_2 use the hg msg type
    mode_pr_ = Mode::PR;
    mode_machine_ = 0;

    ChannelFactory::Instance()->Init(0);
    // create publisher
    lowstate_publisher_ = std::make_unique<ChannelPublisher<LowState_>>("rt/lowstate");
    lowstate_publisher_->InitChannel();

    StartPublishThread();
    
    // // create subscriber
    // lowcmd_subscriber_ = std::make_unique<ChannelSubscriber<LowCmd_>>("rt/lowcmd");
    // lowcmd_subscriber_->InitChannel(
    //     LowCmdHgHandler,
    //     1
    // );

    lowcmd_subscriber_ = std::make_unique<ChannelSubscriber<LowCmd_>>("rt/lowcmd");
    lowcmd_subscriber_->InitChannel(
        [](const void *message) {
            LowCmdHgHandler(message);
        },
        1
    );

    while (true)
    {
        // 你的主逻辑
        std::cout << "22222" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }

    StopPublishThread();

    return 0;
}
