/**
 * This code is wirtten by Cursor, then debuged by me.
 * Cursor writes better and much more faster...
 *
 * Dknt 2024.11
 */

#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/JointState.h>
#include <yaml-cpp/yaml.h>

#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "common.h"
#include "hdf5_recorder.h"

class RosRecorder {
 public:
  explicit RosRecorder(ros::NodeHandle& nh)
      : recorder_(14),  // Changed to 14D for two arms
        running_(false),
        save_thread_(nullptr) {

    std::string rgb_high_topic = nh.param<std::string>("rgb_high_topic", "/camera/high/image_raw");
    std::string rgb_left_topic = nh.param<std::string>("rgb_left_topic", "/camera/left_wrist/image_raw");
    std::string rgb_right_topic = nh.param<std::string>("rgb_right_topic", "/camera/right_wrist/image_raw");
    std::string left_joint_states_topic =
        nh.param<std::string>("left_joint_states_topic", "/left_arm/joint_states");
    std::string right_joint_states_topic =
        nh.param<std::string>("right_joint_states_topic", "/right_arm/joint_states");

    double camera_fps = nh.param<double>("camera_fps", 30);
    time_sync_threshold_ = 1.0 / camera_fps;

    ROS_INFO("rgb_high_topic: %s", rgb_high_topic.c_str());

    // Subscribe to image topics
    sub_high_ = nh.subscribe<sensor_msgs::Image>(rgb_high_topic, 10,
                             &RosRecorder::HighImageCallback, this);
    sub_left_ = nh.subscribe<sensor_msgs::Image>(rgb_left_topic, 10,
                             &RosRecorder::LeftWristImageCallback, this);
    sub_right_ = nh.subscribe<sensor_msgs::Image> (rgb_right_topic, 10,
                              &RosRecorder::RightWristImageCallback, this);

    // Subscribe to joint states for both arms
    sub_joints_left_ = nh.subscribe<sensor_msgs::JointState>(left_joint_states_topic, 10,
                                 &RosRecorder::JointStateLeftCallback, this);
    sub_joints_right_ = nh.subscribe<sensor_msgs::JointState>(right_joint_states_topic, 10,
                                 &RosRecorder::JointStateRightCallback, this);

    ROS_INFO("ROS Recorder initialized for dual arms");
  }

  ~RosRecorder() { StopRecording(); }

  void StartRecording() {
    running_ = true;
    // Clear buffers
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      while (!high_image_buffer_.empty()) {
        high_image_buffer_.pop();
      }
      while (!left_wrist_image_buffer_.empty()) {
        left_wrist_image_buffer_.pop();
      }
      while (!right_wrist_image_buffer_.empty()) {
        right_wrist_image_buffer_.pop();
      }
      while (!joint_state_buffer_left_.empty()) {
        joint_state_buffer_left_.pop();
      }
      while (!joint_state_buffer_right_.empty()) {
        joint_state_buffer_right_.pop();
      }
    }
    save_thread_ =
        std::make_unique<std::thread>(&RosRecorder::SaveThread, this);
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
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg);
    std::lock_guard<std::mutex> lock(data_mutex_);
    high_image_buffer_.push({msg->header.stamp, cv_ptr->image.clone()});
  }

  void LeftWristImageCallback(const sensor_msgs::Image::ConstPtr& msg) {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg);
    std::lock_guard<std::mutex> lock(data_mutex_);
    left_wrist_image_buffer_.push({msg->header.stamp, cv_ptr->image.clone()});
  }

  void RightWristImageCallback(const sensor_msgs::Image::ConstPtr& msg) {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg);
    std::lock_guard<std::mutex> lock(data_mutex_);
    right_wrist_image_buffer_.push({msg->header.stamp, cv_ptr->image.clone()});
  }

  // Add interpolation helper function
  Eigen::MatrixXd InterpolateJointState(const TimestampedJointState& before,
                                        const TimestampedJointState& after,
                                        const ros::Time& target_time) {
    double alpha = (target_time - before.timestamp).toSec() /
                   (after.timestamp - before.timestamp).toSec();
    return before.position * (1.0 - alpha) + after.position * alpha;
  }

  void JointStateLeftCallback(const sensor_msgs::JointState::ConstPtr& msg) {
    Eigen::MatrixXd position(1, 7);
    for (size_t i = 0; i < 7; ++i) {
      position(0, i) = msg->position[i];
    }

    std::lock_guard<std::mutex> lock(data_mutex_);
    joint_state_buffer_left_.push({msg->header.stamp, position, {}, {}});
  }

  void JointStateRightCallback(const sensor_msgs::JointState::ConstPtr& msg) {
    Eigen::MatrixXd position(1, 7);
    for (size_t i = 0; i < 7; ++i) {
      position(0, i) = msg->position[i];
    }

    std::lock_guard<std::mutex> lock(data_mutex_);
    joint_state_buffer_right_.push({msg->header.stamp, position, {}, {}});
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

    while (ros::ok() && running_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));

      std::lock_guard<std::mutex> lock(data_mutex_);
      if (high_image_buffer_.empty() || left_wrist_image_buffer_.empty() ||
           right_wrist_image_buffer_.empty()) {

        continue;
      }

      double high_time = high_image_buffer_.front().timestamp.toSec();
      double left_time = left_wrist_image_buffer_.front().timestamp.toSec();
      double right_time = right_wrist_image_buffer_.front().timestamp.toSec();

      double max_time = std::max({high_time, left_time, right_time});

      if (max_time - high_time > time_sync_threshold_) {
        high_image_buffer_.pop();
        ROS_INFO("Drop high image");
        continue;
      }
      else if (max_time - left_time > time_sync_threshold_) {
        left_wrist_image_buffer_.pop();
        ROS_INFO("Drop left wrist image");
        continue;
      }
      else if (max_time - right_time > time_sync_threshold_) {
        right_wrist_image_buffer_.pop();
        ROS_INFO("Drop right wrist image");
        continue;
      }
      double image_time = max_time;

      // Find surrounding joint states for interpolation
      if (joint_state_buffer_left_.back().timestamp.toSec() < image_time) {
        while (!joint_state_buffer_left_.empty()) {
          joint_state_buffer_left_.pop();
        }
        continue;
      }
      if (joint_state_buffer_right_.back().timestamp.toSec() < image_time) {
        while (!joint_state_buffer_right_.empty()) {
          joint_state_buffer_right_.pop();
        }
        continue;
      }

      auto it_left_before = joint_state_buffer_left_.front();
      auto it_left_after = joint_state_buffer_left_.front();
      auto it_right_before = joint_state_buffer_right_.front();
      auto it_right_after = joint_state_buffer_right_.front();

      while (it_left_after.timestamp.toSec() < image_time) {
        joint_state_buffer_left_.pop();
        if (joint_state_buffer_left_.empty()) {
          ROS_ERROR("No left joint state before image time");
          continue;
        }
        it_left_before = it_left_after;
        it_left_after = joint_state_buffer_left_.front();
      }
      while (it_right_after.timestamp.toSec() < image_time) {
        joint_state_buffer_right_.pop();
        if (joint_state_buffer_right_.empty()) {
          ROS_ERROR("No right joint state before image time");
          continue;
        }
        it_right_before = it_right_after;
        it_right_after = joint_state_buffer_right_.front();
      }

      // Interpolate joint states to image timestamp
      // Eigen::MatrixXd interpolated_joints_left =
      //     InterpolateJointState(it_left_before, it_left_after, ros::Time(image_time));
      // Eigen::MatrixXd interpolated_joints_right =
      //     InterpolateJointState(it_right_before, it_right_after, ros::Time(image_time));

      // Combine joint states
      Eigen::MatrixXd combined_joints(1, 14);
      combined_joints << it_left_before.position, it_right_before.position;

      // Save synchronized data
      cv::Mat high_image=high_image_buffer_.front().image.clone();
      recorder_.PushImgHigh(high_image);
      auto left_wrist_image = left_wrist_image_buffer_.front().image.clone();
      recorder_.PushImgLeftWrist(left_wrist_image);
      auto right_wrist_image = right_wrist_image_buffer_.front().image.clone();
      recorder_.PushImgRightWrist(right_wrist_image);
      recorder_.PushQPos(combined_joints);

      high_image_buffer_.pop();
      left_wrist_image_buffer_.pop();
      right_wrist_image_buffer_.pop();
    }
  }

  static constexpr size_t kBufferSize = 1000;
  static constexpr double kTimeSyncThreshold = 0.01;  // 10ms
  static constexpr int kSaveFrequency = 30;           // Hz
  double time_sync_threshold_;

  Hdf5Recorder recorder_;
  std::atomic_bool running_;
  std::unique_ptr<std::thread> save_thread_;

  // Message buffers
  std::queue<TimestampedImage> high_image_buffer_;
  std::queue<TimestampedImage> left_wrist_image_buffer_;
  std::queue<TimestampedImage> right_wrist_image_buffer_;
  std::queue<TimestampedJointState> joint_state_buffer_left_;
  std::queue<TimestampedJointState> joint_state_buffer_right_;

  ros::Subscriber sub_high_;
  ros::Subscriber sub_left_;
  ros::Subscriber sub_right_;
  ros::Subscriber sub_joints_left_;
  ros::Subscriber sub_joints_right_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "ros_recorder");
  ros::NodeHandle nh;

  RosRecorder recorder(nh);

  recorder.SetInstruction("Example recording");

  recorder.StartRecording();

  ros::Time start_time = ros::Time::now();
  while (ros::ok()) {
    ros::spinOnce();

    ros::Time current_time = ros::Time::now();
    if (current_time - start_time > ros::Duration(1.0)) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  recorder.StopRecording();

  recorder.Save("/home/dknt/Project/rdt/rdt_ws/src/manipulation_recorder/data/test.hdf5");

  return 0;
}
