#include "hdf5_recorder.h"

Hdf5Recorder::Hdf5Recorder(size_t q_size) : q_size_(q_size), num_record_(0) {
  //
}

void Hdf5Recorder::SetInstruction(std::string const ins) {
  this->instruction_ = ins;
}

void Hdf5Recorder::PushImgHigh(
    cv::Mat const& color,
    std::optional<std::reference_wrapper<cv::Mat const>> depth) {
  this->list_color_high_.push_back(color.clone());
  if (depth.has_value()) {
    this->list_depth_high_.push_back(depth.value().get().clone());
  }
  this->num_record_ = std::max(this->num_record_, list_color_high_.size());
}

void Hdf5Recorder::PushImgLeftWrist(
    cv::Mat const& color,
    std::optional<std::reference_wrapper<cv::Mat const>> depth) {
  this->list_color_left_wrist_.push_back(color.clone());
  if (depth.has_value()) {
    this->list_depth_left_wrist_.push_back(depth.value().get().clone());
  }
  this->num_record_ =
      std::max(this->num_record_, list_color_left_wrist_.size());
}

void Hdf5Recorder::PushImgRightWrist(
    cv::Mat const& color,
    std::optional<std::reference_wrapper<cv::Mat const>> depth) {
  this->list_color_right_wrist_.push_back(color.clone());
  if (depth.has_value()) {
    this->list_depth_right_wrist_.push_back(depth.value().get().clone());
  }
  this->num_record_ =
      std::max(this->num_record_, list_color_right_wrist_.size());
}

void Hdf5Recorder::PushAction(Eigen::MatrixXd const& mat) {
  this->list_action_.push_back(mat);
  this->num_record_ = std::max(this->num_record_, list_action_.size());
}

void Hdf5Recorder::PushEffort(Eigen::MatrixXd const& mat) {
  this->list_effort_.push_back(mat);
  this->num_record_ = std::max(this->num_record_, list_effort_.size());
}

void Hdf5Recorder::PushQPos(Eigen::MatrixXd const& mat) {
  this->list_qpos_.push_back(mat);
  this->num_record_ = std::max(this->num_record_, list_qpos_.size());
}

void Hdf5Recorder::PushQVel(Eigen::MatrixXd const& mat) {
  this->list_qvel_.push_back(mat);
  this->num_record_ = std::max(this->num_record_, list_qvel_.size());
}

bool Hdf5Recorder::SaveToFile(std::string const& path) {
  // Check size
  if (this->num_record_ == this->list_color_high_.size()) {
    std::cout << "Saving color image high." << std::endl;
  } else {
    std::cout << "Color image high is not saved." << std::endl;
  }
  if (this->num_record_ == this->list_color_left_wrist_.size()) {
    std::cout << "Saving color image left wrist." << std::endl;
  } else {
    std::cout << "Color image left wrist is not saved." << std::endl;
  }
  if (this->num_record_ == this->list_color_right_wrist_.size()) {
    std::cout << "Saving color image right wrist." << std::endl;
  } else {
    std::cout << "Color image right wrist is not saved." << std::endl;
  }
  if (this->num_record_ == this->list_qpos_.size()) {
    std::cout << "Saving qpos." << std::endl;
  } else {
    std::cout << "qpos is not saved." << std::endl;
  }
  std::cout << "Saving to file: " << path << std::endl;

  /// Save to file
  H5File h5_file(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  Group observations = h5_file.createGroup("observations");
  Group images = observations.createGroup("images");
  Group images_depth = observations.createGroup("images_depth");

  // Save image
  if (this->list_color_high_.size() == this->num_record_)
    ImageBufferToDataSet(images, "cam_high", this->list_color_high_);
  if (this->list_color_left_wrist_.size() == this->num_record_)
    ImageBufferToDataSet(images, "cam_left_wrist",
                         this->list_color_left_wrist_);
  if (this->list_color_right_wrist_.size() == this->num_record_)
    ImageBufferToDataSet(images, "cam_right_wrist",
                         this->list_color_right_wrist_);
  if (this->list_depth_high_.size() == this->num_record_)
    ImageBufferToDataSet(images_depth, "cam_high", this->list_depth_high_);
  if (this->list_depth_left_wrist_.size() == this->num_record_)
    ImageBufferToDataSet(images_depth, "cam_left_wrist",
                         this->list_depth_left_wrist_);
  if (this->list_depth_right_wrist_.size() == this->num_record_)
    ImageBufferToDataSet(images_depth, "cam_right_wrist",
                         this->list_depth_right_wrist_);

  // Save vector
  if (this->list_action_.size() == this->num_record_)
    VectorBufferToDataSet(h5_file, "action", this->list_action_);
  if (this->list_effort_.size() == this->num_record_)
    VectorBufferToDataSet(observations, "effort", this->list_effort_);
  if (this->list_qpos_.size() == this->num_record_)
    VectorBufferToDataSet(observations, "qpos", this->list_qpos_);
  if (this->list_qvel_.size() == this->num_record_)
    VectorBufferToDataSet(observations, "qvel", this->list_qvel_);

  // Save instruction
  hsize_t one_dim[1] = {1};
  DataSpace data_space_one_dim(1, one_dim);
  StrType var_str_type(PredType::C_S1, H5T_VARIABLE);
  DataSet instruction_dataset =
      h5_file.createDataSet("instruction", var_str_type, data_space_one_dim);
  instruction_dataset.write(this->instruction_, var_str_type);

  return true;
}

bool Hdf5Recorder::ImageBufferToDataSet(Group& base_group,
                                        std::string const dataset_name,
                                        std::list<cv::Mat>& list_img) {
  // Encode images
  std::vector<std::vector<uchar>> vec_img_encoded(this->num_record_);
  std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 95};
  size_t max_string_len = 0;

  for (size_t i = 0; i < this->num_record_; ++i) {
    auto img = list_img.front();
    // Encode the image to JPEG format
    cv::imencode(".jpg", img, vec_img_encoded[i], params);
    max_string_len = std::max(max_string_len, vec_img_encoded[i].size());
    list_img.pop_front();
  }

  // Algin strings using '0' padding
  for (size_t i = 0; i < this->num_record_; ++i) {
    if (vec_img_encoded[i].size() < max_string_len) {
      vec_img_encoded[i].resize(max_string_len, uchar(0));
    }
  }

  // Convert into 1 dim buffer
  std::vector<uchar> buffer(this->num_record_ * max_string_len);
  for (size_t i = 0; i < this->num_record_; ++i) {
    std::memcpy(buffer.data() + i * max_string_len, vec_img_encoded[i].data(),
                max_string_len);
  }

  // Create DataSet
  hsize_t data_dim[1] = {this->num_record_};
  DataSpace color_space(1, data_dim);
  StrType str_type(PredType::C_S1, max_string_len);
  // Set the strpad to H5T_STR_NULLPAD.
  // Otherwise, the image will be read wrongly in Python.
  str_type.setStrpad(H5T_STR_NULLPAD);
  DataSet dataset_one_dim =
      base_group.createDataSet(dataset_name, str_type, color_space, H5P_DEFAULT,
                               H5P_DEFAULT, H5P_DEFAULT);
  dataset_one_dim.write(buffer.data(), str_type);

  return true;
}

bool Hdf5Recorder::VectorBufferToDataSet(
    Group& base_group, std::string const dataset_name,
    std::list<Eigen::MatrixXd>& list_vector) {
  // Use Eigen matrix as data buffer.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix(
      this->num_record_, list_vector.front().size());
  for (size_t i = 0; i < this->num_record_; ++i) {
    Eigen::MatrixXd one_row = list_vector.front();
    matrix.row(i) = one_row;
    list_vector.pop_front();
  }

  // Create DataSet
  hsize_t dims[2];  // Dataset dimensions
  dims[0] = matrix.rows();
  dims[1] = matrix.cols();
  DataSpace data_space(2, dims);
  DataSet dataset = base_group.createDataSet(
      dataset_name, PredType::NATIVE_DOUBLE, data_space);
  dataset.write(matrix.data(), PredType::NATIVE_DOUBLE);

  return true;
}

bool Hdf5Recorder::VectorBufferToDataSet(
    H5File& base_file, std::string const dataset_name,
    std::list<Eigen::MatrixXd>& list_vector) {
  // Use Eigen matrix as data buffer.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix(
      this->num_record_, list_vector.front().size());
  for (size_t i = 0; i < this->num_record_; ++i) {
    Eigen::MatrixXd one_row = list_vector.front();
    matrix.row(i) = one_row;
    list_vector.pop_front();
  }

  // Create DataSet
  hsize_t dims[2];  // Dataset dimensions
  dims[0] = matrix.rows();
  dims[1] = matrix.cols();
  DataSpace data_space(2, dims);
  DataSet dataset = base_file.createDataSet(
      dataset_name, PredType::NATIVE_DOUBLE, data_space);
  dataset.write(matrix.data(), PredType::NATIVE_DOUBLE);

  return true;
}
