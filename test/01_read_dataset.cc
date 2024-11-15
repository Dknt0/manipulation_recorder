/**
 * Try to read the RDT dataset and save the data separately.
 *
 * Dknt 2024.11
 */

#include "hdf5_reader.h"

const std::string file_path =
    "/home/dknt/Project/rdt/hdf5_demo/data/episode_0.hdf5";
    
const std::string save_root_path =
    "/home/dknt/Project/rdt/hdf5_demo/data/episode_0_unpack";

int main(int argc, char** argv) {
  Hdf5Reader reader = Hdf5Reader(file_path);
  reader.UnpackDataset(save_root_path);

  return 0;
}
