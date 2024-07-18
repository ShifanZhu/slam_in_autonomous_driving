//
// Created by Shifan on 2024/07/17.
//

#ifndef SLAM_IN_AUTO_DRIVING_IEKF_HPP
#define SLAM_IN_AUTO_DRIVING_IEKF_HPP

#include "common/eigen_types.h"
#include "common/gnss.h"
#include "common/mocap.h"
#include "common/imu.h"
#include "common/landmarks.h"
#include "common/math_utils.h"
#include "common/nav_state.h"
#include "common/odom.h"

#include <glog/logging.h>
#include <iomanip>

namespace sad {

/**
 * 书本第3章介绍的误差卡尔曼滤波器
 * 可以指定观测GNSS的读数，GNSS应该事先转换到车体坐标系
 *
 * 本书使用69维的IEKF，标量类型可以由S指定，默认取double
 * 变量顺序：R, v, p, d1, d2, ..., bg, ba，与论文对应
 * @tparam S    状态变量的精度，取float或double
 */
template <typename S = double>
class IEKF {
   public:
    /// 类型定义
    using SO3 = Sophus::SO3<S>;                     // 旋转变量类型
    using VecT = Eigen::Matrix<S, 3, 1>;            // 向量类型
    using Vec18T = Eigen::Matrix<S, 18, 1>;         // 18维向量类型
    using Vec69T = Eigen::Matrix<S, 69, 1>;         // 18维向量类型
    using Mat3T = Eigen::Matrix<S, 3, 3>;           // 3x3矩阵类型
    // using MotionNoiseT = Eigen::Matrix<S, 18, 18>;  // 运动噪声类型
    using MotionNoiseT = Eigen::Matrix<S, 69, 69>;  // 运动噪声类型
    using OdomNoiseT = Eigen::Matrix<S, 3, 3>;      // 里程计噪声类型
    using GnssNoiseT = Eigen::Matrix<S, 6, 6>;      // GNSS噪声类型
    using Mat18T = Eigen::Matrix<S, 18, 18>;        // 18维方差类型
    using Mat69T = Eigen::Matrix<S, 69, 69>;        // 69维方差类型
    using NavStateT = NavState<S>;                  // 整体名义状态变量类型

    struct Options {
        Options() = default;

        /// IMU 测量与零偏参数
        double imu_dt_ = 0.01;  // IMU测量间隔
        // NOTE IMU噪声项都为离散时间，不需要再乘dt，可以由初始化器指定IMU噪声
        double gyro_var_ = 1e-5;       // 陀螺测量标准差 (可以通过IMU静止，假设陀螺零偏不动，估计得到)
        double acce_var_ = 1e-2;       // 加计测量标准差 (可以通过IMU静止，假设陀螺零偏不动，估计得到)
        double bias_gyro_var_ = 1e-6;  // 陀螺零偏游走标准差
        double bias_acce_var_ = 1e-4;  // 加计零偏游走标准差

        /// 里程计参数
        double odom_var_ = 0.5;
        double odom_span_ = 0.1;        // 里程计测量间隔
        double wheel_radius_ = 0.155;   // 轮子半径
        double circle_pulse_ = 1024.0;  // 编码器每圈脉冲数

        /// RTK 观测参数
        double gnss_pos_noise_ = 0.1;                   // GNSS位置噪声
        double gnss_height_noise_ = 0.1;                // GNSS高度噪声
        double gnss_ang_noise_ = 1.0 * math::kDEG2RAD;  // GNSS旋转噪声

        /// 其他配置
        bool update_bias_gyro_ = false;  // 是否更新陀螺bias
        bool update_bias_acce_ = false;  // 是否更新加计bias
    };

    /**
     * 初始零偏取零
     */
    IEKF(Options option = Options()) : options_(option) { BuildNoise(option); }

    /**
     * 设置初始条件
     * @param options 噪声项配置
     * @param init_bg 初始零偏 陀螺
     * @param init_ba 初始零偏 加计
     * @param gravity 重力
     */
    void SetInitialConditions(Options options, const VecT& init_bg, const VecT& init_ba,
                              const VecT& gravity = VecT(0, 0, -9.8)) {
        BuildNoise(options);
        options_ = options;
        bg_ = init_bg;
        ba_ = init_ba;
        g_ = gravity;
        cov_ = Mat69T::Identity() * 1e-4;
        p_ = VecT(20, 25, 6);
    }

    /// 使用IMU递推
    bool Predict(const IMU& imu);

    /// 使用轮速计观测
    bool ObserveWheelSpeed(const Odom& odom);

    /// 使用GPS观测
    bool ObserveGps(const GNSS& gnss);

    /// 使用MoCap观测
    bool ObserveMoCap(const MoCap& mocap);

    /// 使用3D landmarks观测
    bool ObserveLandmarks(const Landmarks& landmarks);

    /**
     * 使用SE3进行观测
     * @param pose  观测位姿
     * @param trans_noise 平移噪声
     * @param ang_noise   角度噪声
     * @return
     */
    bool ObserveSE3(const SE3& pose, double trans_noise = 0.1, double ang_noise = 1.0 * math::kDEG2RAD);

    /// accessors
    /// 获取全量状态
    NavStateT GetNominalState() const { return NavStateT(current_time_, R_, p_, v_, bg_, ba_); }

    /// 获取SE3 状态
    SE3 GetNominalSE3() const { return SE3(R_, p_); }

    /// 设置状态X
    void SetX(const NavStated& x, const Vec3d& grav) {
        current_time_ = x.timestamp_;
        R_ = x.R_;
        p_ = x.p_;
        v_ = x.v_;
        bg_ = x.bg_;
        ba_ = x.ba_;
        g_ = grav;
    }

    /// 设置协方差
    void SetCov(const Mat69T& cov) { cov_ = cov; }

    /// 获取重力
    Vec3d GetGravity() const { return g_; }

   private:
    void BuildNoise(const Options& options) {
        double ev = options.acce_var_;
        double et = options.gyro_var_;
        double eg = options.bias_gyro_var_;
        double ea = options.bias_acce_var_;

        double ev2 = ev;  // * ev;
        double et2 = et;  // * et;
        double eg2 = eg;  // * eg;
        double ea2 = ea;  // * ea;

        // 设置过程噪声
        // Q_ is 18*18: position, velocity, rotation, bias_gyro, bias_acce, gravity
        // Q_.diagonal() << 0, 0, 0, ev2, ev2, ev2, et2, et2, et2, eg2, eg2, eg2, ea2, ea2, ea2, 0, 0, 0;

        // Q_ is 69*69: R, v, p, d1, d2, ..., bg, ba
        double landmark_var = 0.1;
        double l2 = landmark_var * landmark_var;
        Q_.diagonal() << et2, et2, et2, ev2, ev2, ev2, 0, 0, 0, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, eg2, eg2, eg2, ea2, ea2, ea2;


        // 设置里程计噪声
        double o2 = options_.odom_var_ * options_.odom_var_;
        odom_noise_.diagonal() << o2, o2, o2;

        // 设置GNSS状态
        double gp2 = options.gnss_pos_noise_ * options.gnss_pos_noise_;
        double gh2 = options.gnss_height_noise_ * options.gnss_height_noise_;
        double ga2 = options.gnss_ang_noise_ * options.gnss_ang_noise_;
        gnss_noise_.diagonal() << gp2, gp2, gh2, ga2, ga2, ga2;
    }

    /// 更新名义状态变量，重置error state
    void UpdateAndReset() {
        // dx is 18*1: position, velocity, rotation, bias_gyro, bias_acce, gravity
        // 3.51c
        p_ += dx_.template block<3, 1>(0, 0);
        v_ += dx_.template block<3, 1>(3, 0);
        R_ = R_ * SO3::exp(dx_.template block<3, 1>(6, 0));

        // If we update bias
        if (options_.update_bias_gyro_) {
            bg_ += dx_.template block<3, 1>(9, 0);
            // LOG(INFO) << "update bg: " << bg_.transpose();
        }

        if (options_.update_bias_acce_) {
            ba_ += dx_.template block<3, 1>(12, 0);
            // LOG(INFO) << "update ba: " << ba_.transpose();
        }

        // todo
        // g_ += dx_.template block<3, 1>(15, 0); //? why we update gravity since its derivetive is zero 3.25f?
        //// LOG(INFO) << "update delta g: " << dx_.template block<3, 1>(15, 0).transpose();

        ProjectCov(); // 3.63
        dx_.setZero(); //? why we set dx to zero?
    }

    /// 对P阵进行投影，参考式(3.63)
    void ProjectCov() {
        Mat69T J = Mat69T::Identity();
        J.template block<3, 3>(6, 6) = Mat3T::Identity() - 0.5 * SO3::hat(dx_.template block<3, 1>(6, 0)); // 3.61
        cov_ = J * cov_ * J.transpose(); // 3.63
    }

    /// 成员变量
    double current_time_ = 0.0;  // 当前时间

    /// 名义状态
    VecT p_ = VecT::Zero();
    VecT v_ = VecT::Zero();
    SO3 R_;
    VecT bg_ = VecT::Zero();
    VecT ba_ = VecT::Zero();
    VecT g_{0, 0, -9.8};

    /// 误差状态
    // dx is 18*1: position, velocity, rotation, bias_gyro, bias_acce, gravity
    // Vec18T dx_ = Vec18T::Zero();
    Vec69T dx_ = Vec69T::Zero();

    /// 协方差阵
    Mat69T cov_ = Mat69T::Identity();

    /// 噪声阵
    MotionNoiseT Q_ = MotionNoiseT::Zero();
    OdomNoiseT odom_noise_ = OdomNoiseT::Zero();
    GnssNoiseT gnss_noise_ = GnssNoiseT::Zero();

    /// 标志位
    bool first_gnss_ = true;  // 是否为第一个gnss数据
    bool first_mocap_ = true;  // 是否为第一个mocap数据

    /// 配置项
    Options options_;
};

using IEKFD = IEKF<double>;
using IEKFF = IEKF<float>;

template <typename S>
bool IEKF<S>::Predict(const IMU& imu) {
    assert(imu.timestamp_ >= current_time_);

    double dt = imu.timestamp_ - current_time_;
    if (dt > (5 * options_.imu_dt_) || dt < 0) {
        // 时间间隔不对，可能是第一个IMU数据，没有历史信息
        LOG(INFO) << "skip this imu because dt_ = " << dt;
        current_time_ = imu.timestamp_;
        return false;
    }

    // nominal state 递推 (3.41)
    VecT new_p = p_ + v_ * dt + 0.5 * (R_ * (imu.acce_ - ba_)) * dt * dt + 0.5 * g_ * dt * dt; // 3.41a
    VecT new_v = v_ + R_ * (imu.acce_ - ba_) * dt + g_ * dt; // 3.41b
    SO3 new_R = R_ * SO3::exp((imu.gyro_ - bg_) * dt); // 3.41c Right multiply because IMU data is in local frame
    // std::cout << "t: " << imu.timestamp_ << " p: " << new_p.transpose() << " v: " << new_v.transpose() << " R: " << new_R.unit_quaternion().coeffs().transpose() << std::endl;
    std::cout << "t: " << imu.timestamp_ << " ba: " << ba_.transpose() << " bg: " << bg_.transpose() << " acc: " << imu.acce_.transpose() << " gyro: " << imu.gyro_.transpose() << std::endl;


    R_ = new_R;
    v_ = new_v;
    p_ = new_p;
    // 其余状态维度不变

    // error state 递推
    // 计算运动过程雅可比矩阵 F，见(3.42 or 3.47)
    // F实际上是稀疏矩阵，也可以不用矩阵形式进行相乘而是写成散装形式(faster)，这里为了教学方便，使用矩阵形式
    Mat69T F = Mat69T::Identity();                                                 // 主对角线
    F.template block<3, 3>(0, 3) = Mat3T::Identity() * dt;                         // p 对 v
    F.template block<3, 3>(3, 6) = -R_.matrix() * SO3::hat(imu.acce_ - ba_) * dt;  // v对theta
    F.template block<3, 3>(3, 12) = -R_.matrix() * dt;                             // v 对 ba
    F.template block<3, 3>(3, 15) = Mat3T::Identity() * dt;                        // v 对 g
    F.template block<3, 3>(6, 6) = SO3::exp(-(imu.gyro_ - bg_) * dt).matrix();     // theta 对 theta
    F.template block<3, 3>(6, 9) = -Mat3T::Identity() * dt;                        // theta 对 bg

    // mean and cov prediction
    dx_ = F * dx_;  // 这行其实没必要算，dx_在重置之后应该为零，因此这步可以跳过或注释掉
    // F需要参与cov部分计算，所以保留
    cov_ = F * cov_.eval() * F.transpose() + Q_; // P_pred: Predicted Covariance (3.48b)
    current_time_ = imu.timestamp_;
    return true;
}

template <typename S>
bool IEKF<S>::ObserveWheelSpeed(const Odom& odom) {
    assert(odom.timestamp_ >= current_time_);
    // odom 修正以及雅可比
    // 使用三维的轮速观测，H为3x18，大部分为零
    Eigen::Matrix<S, 3, 18> H = Eigen::Matrix<S, 3, 18>::Zero();
    H.template block<3, 3>(0, 3) = Mat3T::Identity(); // only have direct observation on velocity

    // 卡尔曼增益
    Eigen::Matrix<S, 18, 3> K = cov_ * H.transpose() * (H * cov_ * H.transpose() + odom_noise_).inverse();

    // velocity obs
    double velo_l = options_.wheel_radius_ * odom.left_pulse_ / options_.circle_pulse_ * 2 * M_PI / options_.odom_span_;
    double velo_r =
        options_.wheel_radius_ * odom.right_pulse_ / options_.circle_pulse_ * 2 * M_PI / options_.odom_span_;
    double average_vel = 0.5 * (velo_l + velo_r);

    VecT vel_odom(average_vel, 0.0, 0.0);
    VecT vel_world = R_ * vel_odom;

    dx_ = K * (vel_world - v_); // 3.68

    // update cov
    cov_ = (Mat69T::Identity() - K * H) * cov_;

    UpdateAndReset();
    return true;
}

template <typename S>
bool IEKF<S>::ObserveGps(const GNSS& gnss) {
    /// GNSS 观测的修正
    assert(gnss.unix_time_ >= current_time_);

    // Store first gnsss pose and time
    if (first_gnss_) {
        R_ = gnss.utm_pose_.so3();
        p_ = gnss.utm_pose_.translation();
        first_gnss_ = false;
        current_time_ = gnss.unix_time_;
        return true;
    }

    assert(gnss.heading_valid_); // RTK heading should be valid
    ObserveSE3(gnss.utm_pose_, options_.gnss_pos_noise_, options_.gnss_ang_noise_); // We can observe SE3 directly
    current_time_ = gnss.unix_time_;

    return true;
}

template <typename S>
bool IEKF<S>::ObserveMoCap(const MoCap& mocap) {
    /// MoCap 观测的修正
    assert(mocap.timestamp_ >= current_time_);

    // Store first mocap pose and time
    if (first_mocap_) {
        R_ = mocap.GetSO3();
        p_ = mocap.position_;
        first_mocap_ = false;
        current_time_ = mocap.timestamp_;
        return true;
    }

    ObserveSE3(mocap.GetSE3(), options_.gnss_pos_noise_, options_.gnss_ang_noise_); // We can observe SE3 directly
    current_time_ = mocap.timestamp_;

    return true;
}

template <typename S>
bool IEKF<S>::ObserveSE3(const SE3& pose, double trans_noise, double ang_noise) {
    /// 观测到的状态：既有旋转，也有平移
    /// 观测状态变量中的p, R，H为6x18，其余为零
    Eigen::Matrix<S, 6, 18> H = Eigen::Matrix<S, 6, 18>::Zero(); // Jacobian of observation model w.r.t. error state
    H.template block<3, 3>(0, 0) = Mat3T::Identity();  // P部分 (3.70)
    H.template block<3, 3>(3, 6) = Mat3T::Identity();  // R部分（3.66)

    // 卡尔曼增益和更新过程

    // V is the covariance matrix of the observation noise (gnss_pos_noise_, gnss_ang_noise_)
    Vec6d noise_vec;
    noise_vec << trans_noise, trans_noise, trans_noise, ang_noise, ang_noise, ang_noise;
    Mat6d V = noise_vec.asDiagonal();

    // cov_ is P_pred
    Eigen::Matrix<S, 18, 6> K = cov_ * H.transpose() * (H * cov_ * H.transpose() + V).inverse(); // 3.51a

    // 更新x和cov
    // z - h(x_pred)
    Vec6d innov = Vec6d::Zero(); // innovation
    innov.template head<3>() = (pose.translation() - p_);          // 平移部分(3.67)
    innov.template tail<3>() = (R_.inverse() * pose.so3()).log();  // 旋转部分(3.67)
    // dx is 18*1: position, velocity, rotation, bias_gyro, bias_acce, gravity
    dx_ = K * innov; // 3.51b
    cov_ = (Mat69T::Identity() - K * H) * cov_;  // Corrected covariance (3.51d)

    this->UpdateAndReset(); // Apply 3.51c
    return true;
}

Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& vec) {
    Eigen::Matrix3d skew;
    skew << 0, -vec(2), vec(1),
        vec(2), 0, -vec(0),
        -vec(1), vec(0), 0;
    return skew;
}

Eigen::MatrixXd Exp_SEK3(const Eigen::VectorXd& v) {
    const double TOLERANCE = 1e-10;
    // Computes the vectorized exponential map for SE_K(3)
    int K = (v.size() - 3) / 3;
    Eigen::MatrixXd X = Eigen::MatrixXd::Identity(3 + K, 3 + K);
    Eigen::Matrix3d R;
    Eigen::Matrix3d Jl;
    Eigen::Vector3d w = v.head(3);
    double theta = w.norm();
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    if (theta < TOLERANCE) {
        R = I;
        Jl = I;
    }
    else {
        Eigen::Matrix3d A = skewSymmetric(w);
        double theta2 = theta * theta;
        double stheta = sin(theta);
        double ctheta = cos(theta);
        double oneMinusCosTheta2 = (1 - ctheta) / (theta2);
        Eigen::Matrix3d A2 = A * A;
        R = I + (stheta / theta) * A + oneMinusCosTheta2 * A2;
        Jl = I + oneMinusCosTheta2 * A + ((theta - stheta) / (theta2 * theta)) * A2;
    }
    X.block<3, 3>(0, 0) = R;
    for (int i = 0; i < K; ++i) {
        X.block<3, 1>(0, 3 + i) = Jl * v.segment<3>(3 + 3 * i);
    }
    return X;
}

template <typename S>
bool IEKF<S>:: ObserveLandmarks(const sad::Landmarks& landmarks) {
    static std::vector<Vec3d> global_landmarks({ {0, 0, 0}, {0, 0, 6.5}, {10, 0, 0}, {10, 0, 6.5}, {10, 10, 0}, {10, 10, 6.5}, {0, 10, 0}, {0, 10, 6.5}, {0, 5, 10}, {10, 5, 10}, {0, 6, 0}, {0, 8, 0}, {0, 8, 5}, {0, 6, 5}, {0, 2, 2.5}, {0, 4, 2.5}, {0, 4, 5}, {0, 2, 5} });
    int numLandmarks = landmarks.landmarks_.size();
    
    // Resize observation matrix and observations vector to accommodate all landmarks
    // H is 54 * 69
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3 * numLandmarks, 3 * numLandmarks + 15);
    for (int i = 0; i < 3 * numLandmarks; i += 3) {
        H.block<3, 3>(i, 6) = -Eigen::Matrix3d::Identity();
        H.block<3, 3>(i, 9+i) = Eigen::Matrix3d::Identity();
    }
    // Eigen::VectorXd observations = Eigen::VectorXd::Zero(3 * numLandmarks);
    // Eigen::VectorXd measurements = Eigen::VectorXd::Zero(3 * numLandmarks);

    // Eigen::MatrixXd R = Eigen::MatrixXd::Identity(3 * numLandmarks, 3 * numLandmarks) * 0.1; // Example value, tune as necessary
    // N is 54 * 54
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(3 * numLandmarks, 3 * numLandmarks);
    for (int i = 0; i < 3 * numLandmarks; i += 3) {
        N.block<3, 3>(i, i) = R_.matrix() * Eigen::Matrix3d::Identity() * 0.1 * R_.matrix().transpose();
    }

    // Z is 54 * 1
    Eigen::VectorXd Z = Eigen::VectorXd::Zero(3 * numLandmarks);
    for (int i = 0; i < 3 * numLandmarks; i += 3) {
        Z.segment<3>(i) = R_.matrix() * landmarks.landmarks_[i / 3].tail<3>() - (global_landmarks[i / 3] - p_);
    }



    int dimX = 23; // state_.dimX();
    int dimTheta = 6; // state_.dimTheta();
    int dimP = 69;

    // Remove bias
    Eigen::MatrixXd Theta = Eigen::Matrix<double, 6, 1>::Zero();
    // cov_.block<6, 6>(dimP - dimTheta, dimP - dimTheta) = 0.0001 * Eigen::Matrix<double, 6, 6>::Identity();
    cov_.block(dimP - dimTheta, dimP - dimTheta, 6, 6) = 0.0001 * Eigen::Matrix<double, 6, 6>::Identity();
    cov_.block(0, dimP - dimTheta, dimP - dimTheta, dimTheta) = Eigen::MatrixXd::Zero(dimP - dimTheta, dimTheta);
    cov_.block(dimP - dimTheta, 0, dimTheta, dimP - dimTheta) = Eigen::MatrixXd::Zero(dimTheta, dimP - dimTheta);



    // Compute Kalman Gain
    Eigen::MatrixXd PHT = cov_ * H.transpose(); // 69*54 = 69*69 * 69*54
    Eigen::MatrixXd Q = H * PHT + N; // Before Eq. 20 // 54*54 = 54*69 * 69*54 + 54*54
    Eigen::MatrixXd K = PHT * Q.inverse(); // Before Eq. 20 // 69*54 = 69*54 * 54*54

    // Compute state correction vector
    Eigen::VectorXd delta = K * Z; // 69*1 = 69*54 * 54*1
    Eigen::MatrixXd dX = Exp_SEK3(delta.segment(0, delta.rows() - dimTheta)); // 18*18
    Eigen::VectorXd dTheta = delta.segment(delta.rows() - dimTheta, dimTheta);

    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(dimX, dimX);
    X.block(0, 0, 3, 3) = R_.matrix();
    X.block(0, 3, 3, 1) = v_;
    X.block(0, 4, 3, 1) = p_;
    for (int i = 0; i < numLandmarks; i++) {
        X.block(0, i+5, 3, 1) = global_landmarks[i];
    }
    // Update state
    Eigen::MatrixXd X_new = dX * X; // Right-Invariant Update // Eq. 19
    Eigen::VectorXd Theta_new = Theta + dTheta;

    // update state
    // 变量顺序：R, v, p, d1, d2, ..., bg, ba，与论文对应
    R_ = SO3(X.block(0, 0, 3, 3));
    v_ = X.block(0, 3, 3, 1);
    p_ = X.block(0, 4, 3, 1);


    // Update Covariance
    Eigen::MatrixXd IKH = Eigen::MatrixXd::Identity(dimP, dimP) - K * H; // Eq. 19 // 69*69 = 69*69 - 69*54 * 54*69
    cov_ = IKH * cov_ * IKH.transpose() + K * N * K.transpose(); // Joseph update form // Eq. 19 // 69*69 = 69*69 * 69*69 * 69*69 + 69*54 * 54*54 * 54*69


    // this->UpdateAndReset(); // Apply 3.51c

    return true;
}

}  // namespace sad

#endif  // SLAM_IN_AUTO_DRIVING_IEKF_HPP