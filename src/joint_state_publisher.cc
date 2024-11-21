/**
 * This is to publish the joint states of ARIBOT dual-arm robot.
 *
 * Dknt 2024.11
 */

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <yaml-cpp/yaml.h>

// #include "airbot-driver.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "joint_state_publisher");
  ros::NodeHandle nh;

  std::string right_topic = nh.param<std::string>("right_joint_states_topic",
                                                  "/left_arm/joint_states");
  std::string left_topic = nh.param<std::string>("left_joint_states_topic",
                                                 "/right_arm/joint_states");

  ros::Publisher pub_right =
      nh.advertise<sensor_msgs::JointState>(right_topic, 10);
  ros::Publisher pub_left =
      nh.advertise<sensor_msgs::JointState>(left_topic, 10);
  ros::Rate rate(100);

  // Initialize the Airbot driver

  while (ros::ok()) {
    sensor_msgs::JointState msg_right;
    sensor_msgs::JointState msg_left;

    msg_right.header.stamp = ros::Time::now();
    msg_left.header.stamp = ros::Time::now();
    msg_left.position.resize(7);
    msg_right.position.resize(7); 
    // Get joint states from Airbot driver
    for (size_t i = 0; i < 6; ++i) {
      msg_right.position[i] = 0.0;
      msg_left.position[i] = 0.0;
    }
    msg_right.position[6] = 1.0;  // End effector
    msg_left.position[6] = 1.0;   // End effector


    pub_right.publish(msg_right);
    pub_left.publish(msg_left);
    rate.sleep();
  }

  return 0;
}
