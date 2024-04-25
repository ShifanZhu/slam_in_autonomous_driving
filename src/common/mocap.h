//
// Created by Shifan on 2024/4/18.
//

#ifndef MAPPING_MOCAP_H
#define MAPPING_MOCAP_H

#include <memory>
#include "common/eigen_types.h"

namespace sad {

/// MoCap 读数
struct MoCap {
    MoCap() = default;
    MoCap(double t, const Vec3d& position, const Quatd& orientation) : timestamp_(t), position_(position), orientation_(orientation) {}

    double timestamp_ = 0.0;
    Vec3d position_ = Vec3d::Zero();
    Quatd orientation_ = Quatd(1.0, 0.0, 0.0, 0.0);

    SO3 GetSO3() const { return SO3(orientation_); }
    SE3 GetSE3() const { return SE3(GetSO3(), position_); }
    void SetSE3(const SE3& se3) {
        position_ = se3.translation();
        orientation_ = se3.so3().unit_quaternion();
    }
    // so3 Getso3() const { return SO3().log(); }
    // se3 Getse3() const { return SE3().log(); }
};
}  // namespace sad

using MoCapPtr = std::shared_ptr<sad::MoCap>;

#endif  // MAPPING_MOCAP_H
