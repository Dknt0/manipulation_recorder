/**
 * Passed
 * 
 * Dknt 2024.11
*/

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

int main() {
  H5File h5_file_ =
      H5File("/home/dknt/Project/rdt/hdf5_demo/data/episode_0_test.hdf5", H5F_ACC_RDONLY);
  H5::DataSet img_set = h5_file_.openDataSet("colorset");

  DataSpace data_space = img_set.getSpace();
  hsize_t dims[1];  // dims[0] is the number of images
  int rank = data_space.getSimpleExtentDims(dims);

  std::cout << "Test1" << std::endl;

  StrType str_type = img_set.getStrType();
  size_t string_length = str_type.getSize();
  std::vector<char> buffer(string_length * dims[0]);
  img_set.read(buffer.data(), str_type);

  std::cout << "Test2" << std::endl;

  {
    std::vector<char> img_vector(buffer.begin(),
                                 buffer.begin() + string_length);
    cv::Mat img_row = cv::imdecode(img_vector, cv::IMREAD_COLOR);
    cv::imshow("test", img_row);
    cv::waitKey(0);
  }

  {
    std::vector<char> img_vector(
        buffer.begin() + string_length,
        buffer.begin() + string_length + string_length);
    cv::Mat img_row = cv::imdecode(img_vector, cv::IMREAD_COLOR);
    cv::imshow("test", img_row);
    cv::waitKey(0);
  }

  return 0;
}
