#include <ros/ros.h>
#include <sensor_msgs/JointState.h>

void callback(const sensor_msgs::JointState::ConstPtr& msg) {
  ROS_INFO("Received joint state: %f", msg->position[0]);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "receive_test");
  ros::NodeHandle nh;

  ros::Subscriber sub = nh.subscribe<sensor_msgs::JointState>("/left_arm/joint_states", 10, callback);

  ros::spin();

  return 0;
}
