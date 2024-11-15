#ifndef HDF5_RECORDER_H
#define HDF5_RECORDER_H

#include <H5Cpp.h>
#include <fmt/core.h>

#include <Eigen/Core>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <list>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <string>
#include <thread>

using namespace H5;

class Hdf5Recorder {
 public:
  /// @brief
  /// @param q_size Action size.
  ///   For dual-arm manipulator with two grippers, q_size equals 14.
  Hdf5Recorder(size_t q_size);

  void SetInstruction(std::string const ins);
  void PushImgHigh(
      cv::Mat const& color,
      std::optional<std::reference_wrapper<cv::Mat const>> depth = {});
  void PushImgRightWrist(
      cv::Mat const& color,
      std::optional<std::reference_wrapper<cv::Mat const>> depth = {});
  void PushImgLeftWrist(
      cv::Mat const& color,
      std::optional<std::reference_wrapper<cv::Mat const>> depth = {});
  void PushAction(Eigen::MatrixXd const& mat);
  void PushEffort(Eigen::MatrixXd const& mat);
  void PushQPos(Eigen::MatrixXd const& mat);
  void PushQVel(Eigen::MatrixXd const& mat);

  bool SaveToFile(std::string const& path);

 private:
  bool ImageBufferToDataSet(Group& base_group, std::string const dataset_name,
                            std::list<cv::Mat>& list_img);
  bool VectorBufferToDataSet(Group& base_group, std::string const dataset_name,
                             std::list<Eigen::MatrixXd>& list_vector);
  bool VectorBufferToDataSet(H5File& base_file, std::string const dataset_name,
                             std::list<Eigen::MatrixXd>& list_vector);

 private:
  size_t q_size_;

  size_t num_record_;
  std::string instruction_;
  // Images
  std::list<cv::Mat> list_color_high_;
  std::list<cv::Mat> list_color_left_wrist_;
  std::list<cv::Mat> list_color_right_wrist_;
  std::list<cv::Mat> list_depth_high_;
  std::list<cv::Mat> list_depth_left_wrist_;
  std::list<cv::Mat> list_depth_right_wrist_;
  // Vectors
  std::list<Eigen::MatrixXd> list_action_;
  std::list<Eigen::MatrixXd> list_effort_;
  std::list<Eigen::MatrixXd> list_qpos_;
  std::list<Eigen::MatrixXd> list_qvel_;
};

#endif
