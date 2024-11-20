/**
 * This code is wirtten by Cursor, not me. Cursor writes better and much more faster than me...
 * 
 * Dknt 2024.11
*/

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/JointState.h>
#include <cv_bridge/cv_bridge.h>

#include "hdf5_recorder.h"

#include <yaml-cpp/yaml.h>
#include "common.h"
#include <queue>
#include <mutex>
#include <thread>
#include <memory>

class RosRecorder {
 public: 
  explicit RosRecorder(ros::NodeHandle& nh) 
      : recorder_(14),  // Changed to 14D for two arms
        running_(false),
        save_thread_(nullptr) {
    // Subscribe to image topics
    sub_high_ = nh.subscribe("/camera/high/image_raw", 1, 
        &RosRecorder::HighImageCallback, this);
    sub_left_ = nh.subscribe("/camera/left_wrist/image_raw", 1, 
        &RosRecorder::LeftWristImageCallback, this);
    sub_right_ = nh.subscribe("/camera/right_wrist/image_raw", 1, 
        &RosRecorder::RightWristImageCallback, this);
    
    // Subscribe to joint states for both arms
    sub_joints_1_ = nh.subscribe("/arm1/joint_states", 1, 
        &RosRecorder::JointState1Callback, this);
    sub_joints_2_ = nh.subscribe("/arm2/joint_states", 1, 
        &RosRecorder::JointState2Callback, this);

    ROS_INFO("ROS Recorder initialized for dual arms");
  }

  ~RosRecorder() {
    StopRecording();
  }

  void StartRecording() {
    running_ = true;
    save_thread_ = std::make_unique<std::thread>(&RosRecorder::SaveThread, this);
    ROS_INFO("Started recording thread");
  }

  void StopRecording() {
    if (running_) {
      running_ = false;
      if (save_thread_ && save_thread_->joinable()) {
        save_thread_->join();
      }
      ROS_INFO("Stopped recording thread");
    }
  }

  void Save(const std::string& filepath) {
    if (recorder_.SaveToFile(filepath)) {
      ROS_INFO("Successfully saved recording to: %s", filepath.c_str());
    } else {
      ROS_ERROR("Failed to save recording");
    }
  }

  void SetInstruction(const std::string& instruction) {
    recorder_.SetInstruction(instruction);
  }

 private:
  struct TimestampedImage {
    ros::Time timestamp;
    cv::Mat image;
  };

  struct TimestampedJointState {
    ros::Time timestamp;
    Eigen::MatrixXd position;
    Eigen::MatrixXd velocity;
    Eigen::MatrixXd effort;
  };

  std::mutex data_mutex_;  // Single mutex for all buffer operations

  void HighImageCallback(const sensor_msgs::Image::ConstPtr& msg) {
    try {
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg);
      std::lock_guard<std::mutex> lock(data_mutex_);
      high_image_buffer_.push({msg->header.stamp, cv_ptr->image});
    } catch (const cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }
  }

  void LeftWristImageCallback(const sensor_msgs::Image::ConstPtr& msg) {
    try {
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg);
      std::lock_guard<std::mutex> lock(data_mutex_);
      left_wrist_image_buffer_.push({msg->header.stamp, cv_ptr->image});
    } catch (const cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }
  }

  void RightWristImageCallback(const sensor_msgs::Image::ConstPtr& msg) {
    try {
      cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg);
      std::lock_guard<std::mutex> lock(data_mutex_);
      right_wrist_image_buffer_.push({msg->header.stamp, cv_ptr->image});
    } catch (const cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }
  }

  // Add interpolation helper function
  Eigen::MatrixXd InterpolateJointState(
      const TimestampedJointState& before,
      const TimestampedJointState& after,
      const ros::Time& target_time) {
    double alpha = (target_time - before.timestamp).toSec() / 
                  (after.timestamp - before.timestamp).toSec();
    return before.position * (1.0 - alpha) + after.position * alpha;
  }

  void JointState1Callback(const sensor_msgs::JointState::ConstPtr& msg) {
    Eigen::MatrixXd position(1, 7);
    Eigen::MatrixXd velocity(1, 7);
    Eigen::MatrixXd effort(1, 7);

    for (size_t i = 0; i < 7; ++i) {
      position(0, i) = msg->position[i];
      velocity(0, i) = msg->velocity[i];
      effort(0, i) = msg->effort[i];
    }

    std::lock_guard<std::mutex> lock(data_mutex_);
    joint_state_buffer_1_.push({msg->header.stamp, position, velocity, effort});
  }

  void JointState2Callback(const sensor_msgs::JointState::ConstPtr& msg) {
    Eigen::MatrixXd position(1, 7);
    Eigen::MatrixXd velocity(1, 7);
    Eigen::MatrixXd effort(1, 7);

    for (size_t i = 0; i < 7; ++i) {
      position(0, i) = msg->position[i];
      velocity(0, i) = msg->velocity[i];
      effort(0, i) = msg->effort[i];
    }

    std::lock_guard<std::mutex> lock(data_mutex_);
    joint_state_buffer_2_.push({msg->header.stamp, position, velocity, effort});
  }

  void SaveThread() {
    ros::Rate rate(kSaveFrequency);  // Hz
    while (running_) {
      ProcessBuffers();
      rate.sleep();
    }
    // Final processing of remaining data
    ProcessBuffers();
  }

  void ProcessBuffers() {
    std::lock_guard<std::mutex> lock(data_mutex_);

    while (!high_image_buffer_.empty() && 
           !joint_state_buffer_1_.empty() && joint_state_buffer_1_.size() >= 2 &&
           !joint_state_buffer_2_.empty() && joint_state_buffer_2_.size() >= 2) {
      
      ros::Time image_time = high_image_buffer_.front().timestamp;
      
      // Find surrounding joint states for interpolation
      auto it1_before = joint_state_buffer_1_.front();
      auto it1_after = joint_state_buffer_1_.front();
      auto it2_before = joint_state_buffer_2_.front();
      auto it2_after = joint_state_buffer_2_.front();

      // Ensure we have joint states before and after the image timestamp
      if (it1_before.timestamp > image_time || it2_before.timestamp > image_time) {
        high_image_buffer_.pop();
        continue;
      }

      // Get next states for interpolation
      if (joint_state_buffer_1_.size() >= 2) {
        joint_state_buffer_1_.pop();
        it1_after = joint_state_buffer_1_.front();
      }
      if (joint_state_buffer_2_.size() >= 2) {
        joint_state_buffer_2_.pop();
        it2_after = joint_state_buffer_2_.front();
      }

      if (it1_after.timestamp < image_time || it2_after.timestamp < image_time) {
        continue;
      }

      // Interpolate joint states to image timestamp
      Eigen::MatrixXd interpolated_joints1 = InterpolateJointState(
          it1_before, it1_after, image_time);
      Eigen::MatrixXd interpolated_joints2 = InterpolateJointState(
          it2_before, it2_after, image_time);

      // Combine joint states
      Eigen::MatrixXd combined_joints(1, 14);
      combined_joints << interpolated_joints1, interpolated_joints2;

      // Save synchronized data
      recorder_.PushImgHigh(high_image_buffer_.front().image);
      recorder_.PushQPos(combined_joints);
      // ... handle velocity and effort similarly ...

      high_image_buffer_.pop();
    }

    // Buffer cleanup
    while (joint_state_buffer_1_.size() > kBufferSize) {
      joint_state_buffer_1_.pop();
    }
    while (joint_state_buffer_2_.size() > kBufferSize) {
      joint_state_buffer_2_.pop();
    }
    // ... cleanup other buffers ...
  }

  static constexpr size_t kBufferSize = 1000;
  static constexpr double kTimeSyncThreshold = 0.01;  // 10ms
  static constexpr int kSaveFrequency = 30;  // Hz

  Hdf5Recorder recorder_;
  bool running_;
  std::unique_ptr<std::thread> save_thread_;

  // Message buffers
  std::queue<TimestampedImage> high_image_buffer_;
  std::queue<TimestampedImage> left_wrist_image_buffer_;
  std::queue<TimestampedImage> right_wrist_image_buffer_;
  std::queue<TimestampedJointState> joint_state_buffer_1_;
  std::queue<TimestampedJointState> joint_state_buffer_2_;

  ros::Subscriber sub_high_;
  ros::Subscriber sub_left_;
  ros::Subscriber sub_right_;
  ros::Subscriber sub_joints_1_;
  ros::Subscriber sub_joints_2_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "ros_recorder");
  ros::NodeHandle nh;

  RosRecorder recorder(nh);
  recorder.SetInstruction("Example recording");
  
  recorder.StartRecording();
  ros::Duration(10.0).sleep();
  recorder.StopRecording();

  recorder.Save("/path/to/output.hdf5");

  return 0;
}
