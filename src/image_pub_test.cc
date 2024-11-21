#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  ros::init(argc, argv, "image_pub_test");
  ros::NodeHandle nh;

  ros::Publisher pub_high =
      nh.advertise<sensor_msgs::Image>("/camera/high/image_raw", 10);
  ros::Publisher pub_left =
      nh.advertise<sensor_msgs::Image>("/camera/left_wrist/image_raw", 10);
  ros::Publisher pub_right =
      nh.advertise<sensor_msgs::Image>("/camera/right_wrist/image_raw", 10);

  cv::Mat high_image = cv::imread(
      "/home/dknt/Dataset/tum_dataset/rgbd_dataset_freiburg1_xyz/rgb/"
      "1305031115.243297.png");
  cv::Mat left_image = cv::imread(
      "/home/dknt/Dataset/tum_dataset/rgbd_dataset_freiburg1_xyz/rgb/"
      "1305031117.479403.png");
  cv::Mat right_image = cv::imread(
      "/home/dknt/Dataset/tum_dataset/rgbd_dataset_freiburg1_xyz/rgb/"
      "1305031122.214959.png");

  ros::Rate loop_rate(30);
  while (ros::ok()) {
    std_msgs::Header head;
    head.stamp = ros::Time::now();
    sensor_msgs::ImagePtr msg_high =
        cv_bridge::CvImage(head, "bgr8", high_image).toImageMsg();
    sensor_msgs::ImagePtr msg_left =
        cv_bridge::CvImage(head, "bgr8", left_image).toImageMsg();
    sensor_msgs::ImagePtr msg_right =
        cv_bridge::CvImage(head, "bgr8", right_image).toImageMsg();

    pub_high.publish(*msg_high);
    pub_left.publish(*msg_left);
    pub_right.publish(*msg_right);

    loop_rate.sleep();
  }

  return 0;
}
