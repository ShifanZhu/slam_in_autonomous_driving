//
// Created by Shifan on 2024/4/18.
//

#ifndef MAPPING_LANDMARKS_H
#define MAPPING_LANDMARKS_H

#include <memory>
#include "common/eigen_types.h"

namespace sad {

/// Feature3D 读数
struct Landmarks {
    Landmarks() = default;
    Landmarks(double t, const std::vector<Vec4d>& landmarks) : timestamp_(t), landmarks_(landmarks) {}

    double timestamp_ = 0.0;
    std::vector<Vec4d> landmarks_ = {};
};
}  // namespace sad

using LandmarksPtr = std::shared_ptr<sad::Landmarks>;

#endif  // MAPPING_LANDMARKS_H
