#include "hdf5_reader.h"

Hdf5Reader::Hdf5Reader(std::string file_path) : file_path_(file_path) {
  std::cout << "Open H5File: " << file_path_ << std::endl;
  h5_file_ = H5File(file_path_, H5F_ACC_RDONLY);
  action_ = h5_file_.openDataSet("action");
  // base_action_ = h5_file_.openDataSet("base_action");
  instruction_ = h5_file_.openDataSet("instruction");
  // Observations
  observations_ = h5_file_.openGroup("observations");
  effort_ = observations_.openDataSet("effort");
  qpos_ = observations_.openDataSet("qpos");
  qvel_ = observations_.openDataSet("qvel");
  // Color images
  images_ = observations_.openGroup("images");
  color_cam_high_ = images_.openDataSet("cam_high");
  color_cam_left_wrist_ = images_.openDataSet("cam_left_wrist");
  color_cam_right_wrist_ = images_.openDataSet("cam_right_wrist");
  // Depth images
  images_depth_ = observations_.openGroup("images_depth");
  depth_cam_high_ = images_depth_.openDataSet("cam_high");
  depth_cam_left_wrist_ = images_depth_.openDataSet("cam_left_wrist");
  depth_cam_right_wrist_ = images_depth_.openDataSet("cam_right_wrist");
}

bool Hdf5Reader::UnpackImageSet(DataSet const& img_set, std::string const path,
                                int flag) {
  // Get number of image
  DataSpace data_space = img_set.getSpace();
  hsize_t dims[1];  // dims[0] is the number of images
  int rank = data_space.getSimpleExtentDims(dims);

  // Get fixed string size
  DataType data_type = img_set.getDataType();
  if (data_type.getClass() != H5T_STRING) {
    std::cerr << "Wrong DataSet type. The image should be encoded as string."
              << std::endl;
    return false;
  }
  StrType str_type = img_set.getStrType();
  if (str_type.isVariableStr()) {
    std::cerr << "Wrong DataSet type. The image string should have fixed size."
              << std::endl;
    return false;
  }
  size_t string_length = str_type.getSize();

  std::cout << "Converting dataset: " << img_set.getObjName() << " with "
            << dims[0] << " images." << std::endl;

  std::vector<char> buffer(string_length * dims[0]);
  img_set.read(buffer.data(), str_type);

  // Check path existence
  std::string save_dir = fmt::format("{}{}", path, img_set.getObjName());
  if (!save_dir.empty() && !std::filesystem::exists(save_dir)) {
    try {
      std::filesystem::create_directories(save_dir);
      std::cout << "Directory created: " << save_dir << std::endl;
    } catch (const std::filesystem::filesystem_error& e) {
      std::cerr << "Error creating directory: " << e.what() << std::endl;
      return false;
    }
  }

  // Save image
  for (size_t i = 0; i < dims[0]; ++i) {
    size_t offset = i * string_length;
    std::vector<char> img_vector(buffer.begin() + offset,
                                 buffer.begin() + offset + string_length);

    cv::Mat img_row = cv::imdecode(img_vector, flag);

    std::string save_path = fmt::format("{}/{:05d}.png", save_dir, i);
    cv::imwrite(save_path, img_row);
  }

  return true;
}

bool Hdf5Reader::UnpackVectorSet(DataSet const& vec_set,
                                 std::string const path) {
  DataSpace data_space = vec_set.getSpace();
  hsize_t dims[2];
  int rank = data_space.getSimpleExtentDims(dims);

  DataType data_type = vec_set.getDataType();
  if (data_type.getClass() != H5T_FLOAT) {
    std::cerr << "Wrong DataSet type. The vector type should be double."
              << std::endl;
    return false;
  }

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix(
      dims[0], dims[1]);

  // Use NATIVE_DOUBLE instead of type derived from DataSet
  vec_set.read(matrix.data(), PredType::NATIVE_DOUBLE);

  // Check path existence
  H5std_string dataset_path = vec_set.getObjName();
  std::size_t last_slash = dataset_path.find_last_of('/');
  std::string save_dir;
  std::string file_name;
  if (last_slash == std::string::npos) {
    save_dir = path;
  } else {
    save_dir = fmt::format("{}{}", path, dataset_path.substr(0, last_slash));
  }
  if (!save_dir.empty() && !std::filesystem::exists(save_dir)) {
    try {
      std::filesystem::create_directories(save_dir);
      std::cout << "Directory created: " << save_dir << std::endl;
    } catch (const std::filesystem::filesystem_error& e) {
      std::cerr << "Error creating directory: " << e.what() << std::endl;
      return false;
    }
  }
  file_name = fmt::format("{}{}.txt", path, dataset_path);

  // Save the matrix
  std::ofstream ofs(file_name);
  ofs << "rows " << matrix.rows() << " cols " << matrix.cols() << std::endl;
  ofs << std::setprecision(std::numeric_limits<double>::max_digits10);
  for (size_t i = 0; i < matrix.rows(); ++i) {
    ofs << matrix.row(i) << std::endl;
  }
  ofs.close();
  std::cout << "Saved file: " << file_name << std::endl;

  return true;
}

bool Hdf5Reader::UnpackInstruction(DataSet const& ins_set,
                                   std::string const path) {
  // Get number of image
  DataSpace data_space = ins_set.getSpace();
  hsize_t dims[1];  // dims[0] is the number of images
  int rank = data_space.getSimpleExtentDims(dims);

  DataType data_type = ins_set.getDataType();
  if (!data_type.getClass() == H5T_STRING) {
    std::cerr << "Wrong DataSet type. The image should be encoded as string."
              << std::endl;
    return false;
  }

  // Read a variable string
  std::vector<char*> buffer(1);
  ins_set.read(buffer.data(), data_type);

  // Save to file
  std::string ins_str(buffer[0]);
  std::string file_name = fmt::format("{}/{}.txt", path, "instruction");
  std::ofstream ofs(file_name);

  ofs << ins_str << std::endl;
  ofs.close();
  std::cout << "Saved file: " << file_name << std::endl;

  return true;
}

bool Hdf5Reader::ShowImageSet(DataSet const& img_set, std::string const w_name,
                              int flag) {
  // Get number of image
  DataSpace data_space = img_set.getSpace();
  hsize_t dims[1];  // dims[0] is the number of images
  int rank = data_space.getSimpleExtentDims(dims);

  // Get fixed string size
  DataType data_type = img_set.getDataType();
  if (!data_type.getClass() == H5T_STRING) {
    std::cerr << "Wrong DataSet type. The image should be encoded as string."
              << std::endl;
    return false;
  }
  StrType str_type = img_set.getStrType();
  if (str_type.isVariableStr()) {
    std::cerr << "Wrong DataSet type. The image string should have fixed size."
              << std::endl;
    return false;
  }
  size_t string_length = str_type.getSize();

  std::cout << "Displaying dataset: " << img_set.getObjName() << " with "
            << dims[0] << " images." << std::endl;

  std::vector<char> buffer(string_length * dims[0]);
  img_set.read(buffer.data(), str_type);

  for (size_t i = 0; i < dims[0]; ++i) {
    size_t offset = i * string_length;
    std::vector<char> img_vector(buffer.begin() + offset,
                                 buffer.begin() + offset + string_length);

    cv::Mat img = cv::imdecode(img_vector, flag);
    if (flag == cv::IMREAD_COLOR)
      cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    else {
      cv::normalize(img, img, 0, 255, cv::NORM_MINMAX, CV_8U);
    }

    cv::imshow(w_name, img);
    cv::waitKey(33);  // 30Hz
  }

  return true;
}
