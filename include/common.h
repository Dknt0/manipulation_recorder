#ifndef MANIPULATION_RECORDER_COMMON_H
#define MANIPULATION_RECORDER_COMMON_H

#include <Eigen/Core>
#include <Eigen/Geometry>

template <typename T>
inline T LinearInterpolation(const T& start, const T& end, const typename T::Scalar& ratio) {
  if constexpr (std::is_same_v<T, Eigen::Quaternion<typename T::Scalar>>) {
    return start.slerp(ratio, end);
  } else {
    return start + (end - start) * ratio;
  }
}

#endif  // MANIPULATION_RECORDER_COMMON_H
