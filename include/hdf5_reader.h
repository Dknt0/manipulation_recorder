/**
 * This class is to read an existing HDF5 dataset, save it as separate images
 * and files.
 *
 * Dknt 2024.11
 */

#ifndef HDF5_READER_H
#define HDF5_READER_H

#include <H5Cpp.h>
#include <fmt/core.h>

#include <Eigen/Core>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <thread>

using namespace H5;

class Hdf5Reader {
 public:
  Hdf5Reader(std::string file_path);

  void ShowDataset() {
    /// TODO: Rewrite this function. Maybe I should add some GUI
    ShowImageSet(color_cam_high_, "test", cv::IMREAD_COLOR);
    ShowImageSet(color_cam_left_wrist_, "test", cv::IMREAD_COLOR);
    ShowImageSet(color_cam_right_wrist_, "test", cv::IMREAD_COLOR);
    ShowImageSet(depth_cam_high_, "test", cv::IMREAD_UNCHANGED);
    ShowImageSet(depth_cam_left_wrist_, "test", cv::IMREAD_UNCHANGED);
    ShowImageSet(depth_cam_right_wrist_, "test", cv::IMREAD_UNCHANGED);
  }

  void UnpackDataset(std::string const path) {
    UnpackImageSet(color_cam_high_, path, cv::IMREAD_COLOR);
    UnpackImageSet(color_cam_left_wrist_, path, cv::IMREAD_COLOR);
    UnpackImageSet(color_cam_right_wrist_, path, cv::IMREAD_COLOR);
    UnpackImageSet(depth_cam_high_, path, cv::IMREAD_UNCHANGED);
    UnpackImageSet(depth_cam_right_wrist_, path, cv::IMREAD_UNCHANGED);
    UnpackImageSet(depth_cam_left_wrist_, path, cv::IMREAD_UNCHANGED);
    UnpackVectorSet(action_, path);
    // UnpackVectorSet(base_action_, path);
    UnpackVectorSet(effort_, path);
    UnpackVectorSet(qpos_, path);
    UnpackVectorSet(qvel_, path);
    UnpackInstruction(instruction_, path);
  }

 private:
  bool UnpackImageSet(DataSet const& img_set, std::string const path,
                      int flag = cv::IMREAD_COLOR);

  bool UnpackVectorSet(DataSet const& vec_set, std::string const path);

  bool UnpackInstruction(DataSet const& ins_set, std::string const path);

  bool ShowImageSet(DataSet const& img_set, std::string const w_name = "test",
                    int flag = cv::IMREAD_COLOR);

 private:
  std::string file_path_;
  H5File h5_file_;

  /// DataSets
  DataSet action_;
  DataSet base_action_;
  DataSet instruction_;
  // Observations
  Group observations_;
  DataSet effort_;
  DataSet qpos_;
  DataSet qvel_;
  // Color images
  Group images_;
  DataSet color_cam_high_;
  DataSet color_cam_left_wrist_;
  DataSet color_cam_right_wrist_;
  // Depth images
  Group images_depth_;
  DataSet depth_cam_high_;
  DataSet depth_cam_left_wrist_;
  DataSet depth_cam_right_wrist_;
};

#endif
