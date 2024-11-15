#include "hdf5_recorder.h"

int main(int argc, char** argv) {
  std::string data_path =
      "/home/dknt/Project/rdt/hdf5_demo/data/episode_0_unpack";

  std::string file_path =
      "/home/dknt/Project/rdt/hdf5_demo/data/episode_0_test.hdf5";

  Hdf5Recorder recoder(14);

  size_t num_step = 148;

  // Set instruction
  std::ifstream ins_ofs(fmt::format("{}/instruction.txt", data_path));
  std::string instruction;
  std::getline(ins_ofs, instruction);
  ins_ofs.close();
  recoder.SetInstruction(instruction);

  // Push Image
  for (size_t i = 0; i < num_step; ++i) {
    std::string color_high_path =
        fmt::format("{}/observations/images/cam_high/{:05d}.png", data_path, i);
    std::string depth_high_path = fmt::format(
        "{}/observations/images_depth/cam_high/{:05d}.png", data_path, i);
    cv::Mat img = cv::imread(color_high_path, cv::IMREAD_COLOR);
    cv::Mat img_depth = cv::imread(depth_high_path, cv::IMREAD_UNCHANGED);
    recoder.PushImgHigh(img, img_depth);

    std::string color_left_wrist_path = fmt::format(
        "{}/observations/images/cam_left_wrist/{:05d}.png", data_path, i);
    std::string depth_left_wrist_path = fmt::format(
        "{}/observations/images_depth/cam_left_wrist/{:05d}.png", data_path, i);
    img = cv::imread(color_left_wrist_path, cv::IMREAD_COLOR);
    img_depth = cv::imread(depth_left_wrist_path, cv::IMREAD_UNCHANGED);
    recoder.PushImgLeftWrist(img, img_depth);

    std::string color_right_wrist_path = fmt::format(
        "{}/observations/images/cam_right_wrist/{:05d}.png", data_path, i);
    std::string depth_right_wrist_path =
        fmt::format("{}/observations/images_depth/cam_right_wrist/{:05d}.png",
                    data_path, i);
    img = cv::imread(color_right_wrist_path, cv::IMREAD_COLOR);
    img_depth = cv::imread(depth_right_wrist_path, cv::IMREAD_UNCHANGED);
    recoder.PushImgRightWrist(img, img_depth);
  }

  // Push Vector
  {
    std::string action_path = fmt::format("{}/action.txt", data_path);
    std::ifstream action_ifs(action_path);

    double temp_double;
    std::string temp_string;
    action_ifs >> temp_string >> temp_double >> temp_string >> temp_double;
    for (size_t i = 0; i < num_step; ++i) {
      Eigen::MatrixXd mat(1, 14);
      for (size_t j = 0; j < 14; j++) {
        action_ifs >> mat(0, j);
      }
      recoder.PushAction(mat);
    }
    action_ifs.close();
  }
  {
    std::string vec_path = fmt::format("{}/observations/effort.txt", data_path);
    std::ifstream vec_ifs(vec_path);

    double temp_double;
    std::string temp_string;
    vec_ifs >> temp_string >> temp_double >> temp_string >> temp_double;
    for (size_t i = 0; i < num_step; ++i) {
      Eigen::MatrixXd mat(1, 14);
      for (size_t j = 0; j < 14; j++) {
        vec_ifs >> mat(0, j);
      }
      recoder.PushEffort(mat);
    }
    vec_ifs.close();
  }
  {
    std::string vec_path = fmt::format("{}/observations/qpos.txt", data_path);
    std::ifstream vec_ifs(vec_path);

    double temp_double;
    std::string temp_string;
    vec_ifs >> temp_string >> temp_double >> temp_string >> temp_double;
    for (size_t i = 0; i < num_step; ++i) {
      Eigen::MatrixXd mat(1, 14);
      for (size_t j = 0; j < 14; j++) {
        vec_ifs >> mat(0, j);
      }
      recoder.PushQPos(mat);
    }
    vec_ifs.close();
  }
  {
    std::string vec_path = fmt::format("{}/observations/qvel.txt", data_path);
    std::ifstream vec_ifs(vec_path);

    double temp_double;
    std::string temp_string;
    vec_ifs >> temp_string >> temp_double >> temp_string >> temp_double;
    for (size_t i = 0; i < num_step; ++i) {
      Eigen::MatrixXd mat(1, 14);
      for (size_t j = 0; j < 14; j++) {
        vec_ifs >> mat(0, j);
      }
      recoder.PushQVel(mat);
    }
    vec_ifs.close();
  }

  // Save to hdf5 file
  recoder.SaveToFile(file_path);

  return 0;
}
