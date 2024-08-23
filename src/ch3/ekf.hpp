//
// Created by Shifan on 2024/07/17.
//

#ifndef SLAM_IN_AUTO_DRIVING_EKF_HPP
#define SLAM_IN_AUTO_DRIVING_EKF_HPP

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
#include <fstream>

namespace sad {

/**
 * 书本第3章介绍的误差卡尔曼滤波器
 * 可以指定观测GNSS的读数，GNSS应该事先转换到车体坐标系
 *
 * 本书使用69维的EKF，标量类型可以由S指定，默认取double
 * 变量顺序：R, v, p, d1, d2, ..., bg, ba，与论文对应
 * @tparam S    状态变量的精度，取float或double
 */
template <typename S = double>
class EKF {
   public:
    /// 类型定义
    using SO3 = Sophus::SO3<S>;                     // 旋转变量类型
    using VecT = Eigen::Matrix<S, 3, 1>;            // 向量类型
    using Vec18T = Eigen::Matrix<S, 18, 1>;         // 18维向量类型
    using Vec69T = Eigen::Matrix<S, 69, 1>;         // 18维向量类型
    using Mat3T = Eigen::Matrix<S, 3, 3>;           // 3x3矩阵类型
    using MotionNoiseT = Eigen::Matrix<S, 18, 18>;  // 运动噪声类型
    // using MotionNoiseT = Eigen::Matrix<S, 69, 69>;  // 运动噪声类型
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
    EKF(Options option = Options()) : options_(option) { BuildNoise(option); }

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
        cov_ = Mat18T::Identity() * 1e-4;
        p_ = VecT(20, 25, 10);
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
    void SetCov(const Mat18T& cov) { cov_ = cov; }

    /// 获取重力
    Vec3d GetGravity() const { return g_; }
    SO3 GetOrientation() const { return R_; }
    void SetOrientation(SO3& R) {R_ = R;}

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
        Q_.diagonal() << 0, 0, 0, ev2, ev2, ev2, et2, et2, et2, eg2, eg2, eg2, ea2, ea2, ea2, 0, 0, 0;

        // Q_ is 69*69: R, v, p, d1, d2, ..., bg, ba
        // double landmark_var = 0.1;
        // double l2 = landmark_var * landmark_var;
        // Q_.diagonal() << et2, et2, et2, ev2, ev2, ev2, 0, 0, 0, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, l2, eg2, eg2, eg2, ea2, ea2, ea2;


        // 设置里程计噪声
        double o2 = options_.odom_var_ * options_.odom_var_;
        odom_noise_.diagonal() << o2, o2, o2;

        // 设置GNSS状态
        double gp2 = options.gnss_pos_noise_ * options.gnss_pos_noise_;
        double gh2 = options.gnss_height_noise_ * options.gnss_height_noise_;
        double ga2 = options.gnss_ang_noise_ * options.gnss_ang_noise_;
        gnss_noise_.diagonal() << gp2, gp2, gh2, ga2, ga2, ga2;
    }

    // /// 更新名义状态变量，重置error state
    // void UpdateAndReset() {
    //     // dx is 18*1: position, velocity, rotation, bias_gyro, bias_acce, gravity
    //     // 3.51c
    //     p_ += dx_.template block<3, 1>(0, 0);
    //     v_ += dx_.template block<3, 1>(3, 0);
    //     R_ = R_ * SO3::exp(dx_.template block<3, 1>(6, 0));

    //     // If we update bias
    //     if (options_.update_bias_gyro_) {
    //         bg_ += dx_.template block<3, 1>(9, 0);
    //         // LOG(INFO) << "update bg: " << bg_.transpose();
    //     }

    //     if (options_.update_bias_acce_) {
    //         ba_ += dx_.template block<3, 1>(12, 0);
    //         // LOG(INFO) << "update ba: " << ba_.transpose();
    //     }

    //     // todo
    //     // g_ += dx_.template block<3, 1>(15, 0); //? why we update gravity since its derivetive is zero 3.25f?
    //     //// LOG(INFO) << "update delta g: " << dx_.template block<3, 1>(15, 0).transpose();

    //     ProjectCov(); // 3.63
    //     dx_.setZero(); //? why we set dx to zero?
    // }

    // /// 对P阵进行投影，参考式(3.63)
    // void ProjectCov() {
    //     Mat18T J = Mat18T::Identity();
    //     J.template block<3, 3>(6, 6) = Mat3T::Identity() - 0.5 * SO3::hat(dx_.template block<3, 1>(6, 0)); // 3.61
    //     cov_ = J * cov_ * J.transpose(); // 3.63
    // }

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
    // Vec69T dx_ = Vec69T::Zero();

    /// 协方差阵
    Mat18T cov_ = Mat18T::Identity();

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

using EKFD = EKF<double>;
using EKFF = EKF<float>;


// const Eigen::MatrixXd Xinv(Eigen::MatrixXd& X) {
//     // const Eigen::MatrixXd Xinv(Eigen::MatrixXd & X) const {
//     int dimX = 23;
//     Eigen::MatrixXd Xinv = Eigen::MatrixXd::Identity(dimX, dimX);
//     Eigen::Matrix3d RT = X.block<3, 3>(0, 0).transpose();
//     Xinv.block<3, 3>(0, 0) = RT;
//     for (int i = 3; i < dimX; ++i) {
//         Xinv.block<3, 1>(0, i) = -RT * X.block<3, 1>(0, i);
//     }
//     return Xinv;
// }

Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& vec) {
    Eigen::Matrix3d skew;
    skew << 0, -vec(2), vec(1),
        vec(2), 0, -vec(0),
        -vec(1), vec(0), 0;
    return skew;
}

// long int factorial(int n) {
//     return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
// }

// Eigen::Matrix3d Gamma_SO3(const Eigen::Vector3d& w, int m) {
//     // Computes mth integral of the exponential map: \Gamma_m = \sum_{n=0}^{\infty} \dfrac{1}{(n+m)!} (w^\wedge)^n
//     assert(m >= 0);
//     Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
//     double theta = w.norm();
//     if (theta < 1e-10) {
//         return (1.0 / factorial(m)) * I; // TODO: There is a better small value approximation for exp() given in Trawny p.19
//     }
//     Eigen::Matrix3d A = skewSymmetric(w);
//     double theta2 = theta * theta;

//     // Closed form solution for the first 3 cases
//     switch (m) {
//     case 0: // Exp map of SO(3) // Taylor expansion: Directly computes the rotation matrix for a given angular velocity vector.
//         return I + (sin(theta) / theta) * A + ((1 - cos(theta)) / theta2) * A * A;

//     case 1: // Left Jacobian of SO(3) // Mapping small angular changes in the Lie algebra to changes in the Lie group
//         // This integral captures how the exponential map smoothly transitions from the identity to ew∧ as t goes from 0 to 1.
//         // exp(w^) = I + w^Jl(w)
//     // eye(3) - A*(1/theta^2) * (R - eye(3) - A);
//     // eye(3) + (1-cos(theta))/theta^2 * A + (theta-sin(theta))/theta^3 * A^2;
//         return I + ((1 - cos(theta)) / theta2) * A + ((theta - sin(theta)) / (theta2 * theta)) * A * A;

//     case 2:
//         // 0.5*eye(3) - (1/theta^2) * (R - eye(3) - A - 0.5*A^2);
//         // 0.5*eye(3) + (theta-sin(theta))/theta^3 * A + (2*(cos(theta)-1) + theta^2)/(2*theta^4) * A^2
//         return 0.5 * I + (theta - sin(theta)) / (theta2 * theta) * A + (theta2 + 2 * cos(theta) - 2) / (2 * theta2 * theta2) * A * A;

//     default: // General case 
//         Eigen::Matrix3d R = I + (sin(theta) / theta) * A + ((1 - cos(theta)) / theta2) * A * A;
//         Eigen::Matrix3d S = I;
//         Eigen::Matrix3d Ak = I;
//         long int kfactorial = 1;
//         for (int k = 1; k <= m; ++k) {
//             kfactorial = kfactorial * k;
//             Ak = (Ak * A).eval();
//             S = (S + (1.0 / kfactorial) * Ak).eval();
//         }
//         if (m == 0) {
//             return R;
//         }
//         else if (m % 2) { // odd 
//             return (1.0 / kfactorial) * I + (pow(-1, (m + 1) / 2) / pow(theta, m + 1)) * A * (R - S);
//         }
//         else { // even
//             return (1.0 / kfactorial) * I + (pow(-1, m / 2) / pow(theta, m)) * (R - S);
//         }
//     }

// }

// Eigen::Matrix3d Exp_SO3(const Eigen::Vector3d& w) {
//     // Computes the vectorized exponential map for SO(3)
//     return Gamma_SO3(w, 0);
// }

// Eigen::Matrix3d LeftJacobian_SO3(const Eigen::Vector3d& w) {
//     // Computes the Left Jacobian of SO(3)
//     return Gamma_SO3(w, 1);
// }

// Eigen::Matrix3d RightJacobian_SO3(const Eigen::Vector3d& w) {
//     // Computes the Right Jacobian of SO(3)
//     return Gamma_SO3(-w, 1);
// }

// // Compute Analytical state transition matrix
// Eigen::MatrixXd StateTransitionMatrix(Eigen::Vector3d& w, Eigen::Vector3d& a, double dt) {
//     Eigen::Vector3d phi = w * dt;
//     Eigen::Matrix3d G0 = Gamma_SO3(phi, 0); // Computation can be sped up by computing G0,G1,G2 all at once
//     Eigen::Matrix3d G1 = Gamma_SO3(phi, 1); // TODO: These are also needed for the mean propagation, we should not compute twice
//     Eigen::Matrix3d G2 = Gamma_SO3(phi, 2);
//     Eigen::Matrix3d G0t = G0.transpose();
//     Eigen::Matrix3d G1t = G1.transpose();
//     Eigen::Matrix3d G2t = G2.transpose();
//     Eigen::Matrix3d G3t = Gamma_SO3(-phi, 3);

//     int dimX = 23;
//     int dimTheta = 6;
//     int dimP = 69;
//     Eigen::MatrixXd Phi = Eigen::MatrixXd::Identity(dimP, dimP);

//     // Compute the complicated bias terms (derived for the left invariant case)
//     Eigen::Matrix3d ax = skewSymmetric(a);
//     Eigen::Matrix3d wx = skewSymmetric(w);
//     Eigen::Matrix3d wx2 = wx * wx;
//     double dt2 = dt * dt;
//     double dt3 = dt2 * dt;
//     double theta = w.norm();
//     double theta2 = theta * theta;
//     double theta3 = theta2 * theta;
//     double theta4 = theta3 * theta;
//     double theta5 = theta4 * theta;
//     double theta6 = theta5 * theta;
//     double theta7 = theta6 * theta;
//     double thetadt = theta * dt;
//     double thetadt2 = thetadt * thetadt;
//     double thetadt3 = thetadt2 * thetadt;
//     double sinthetadt = sin(thetadt);
//     double costhetadt = cos(thetadt);
//     double sin2thetadt = sin(2 * thetadt);
//     double cos2thetadt = cos(2 * thetadt);
//     double thetadtcosthetadt = thetadt * costhetadt;
//     double thetadtsinthetadt = thetadt * sinthetadt;

//     Eigen::Matrix3d Phi25L = G0t * (ax * G2t * dt2
//         + ((sinthetadt - thetadtcosthetadt) / (theta3)) * (wx * ax)
//         - ((cos2thetadt - 4 * costhetadt + 3) / (4 * theta4)) * (wx * ax * wx)
//         + ((4 * sinthetadt + sin2thetadt - 4 * thetadtcosthetadt - 2 * thetadt) / (4 * theta5)) * (wx * ax * wx2)
//         + ((thetadt2 - 2 * thetadtsinthetadt - 2 * costhetadt + 2) / (2 * theta4)) * (wx2 * ax)
//         - ((6 * thetadt - 8 * sinthetadt + sin2thetadt) / (4 * theta5)) * (wx2 * ax * wx)
//         + ((2 * thetadt2 - 4 * thetadtsinthetadt - cos2thetadt + 1) / (4 * theta6)) * (wx2 * ax * wx2));

//     Eigen::Matrix3d Phi35L = G0t * (ax * G3t * dt3
//         - ((thetadtsinthetadt + 2 * costhetadt - 2) / (theta4)) * (wx * ax)
//         - ((6 * thetadt - 8 * sinthetadt + sin2thetadt) / (8 * theta5)) * (wx * ax * wx)
//         - ((2 * thetadt2 + 8 * thetadtsinthetadt + 16 * costhetadt + cos2thetadt - 17) / (8 * theta6)) * (wx * ax * wx2)
//         + ((thetadt3 + 6 * thetadt - 12 * sinthetadt + 6 * thetadtcosthetadt) / (6 * theta5)) * (wx2 * ax)
//         - ((6 * thetadt2 + 16 * costhetadt - cos2thetadt - 15) / (8 * theta6)) * (wx2 * ax * wx)
//         + ((4 * thetadt3 + 6 * thetadt - 24 * sinthetadt - 3 * sin2thetadt + 24 * thetadtcosthetadt) / (24 * theta7)) * (wx2 * ax * wx2));


//     // TODO: Get better approximation using taylor series when theta < tol
//     const double tol = 1e-6;
//     if (theta < tol) {
//         Phi25L = (1 / 2) * ax * dt2;
//         Phi35L = (1 / 6) * ax * dt3;
//     }

//     // Fill out analytical state transition matrices
//     // Compute left-invariant state transisition matrix
//     Phi.block<3, 3>(0, 0) = G0t; // Phi_11
//     Phi.block<3, 3>(3, 0) = -G0t * skewSymmetric(G1 * a) * dt; // Phi_21
//     Phi.block<3, 3>(6, 0) = -G0t * skewSymmetric(G2 * a) * dt2; // Phi_31
//     Phi.block<3, 3>(3, 3) = G0t; // Phi_22
//     Phi.block<3, 3>(6, 3) = G0t * dt; // Phi_32
//     Phi.block<3, 3>(6, 6) = G0t; // Phi_33
//     for (int i = 5; i < dimX; ++i) {
//         Phi.block<3, 3>((i - 2) * 3, (i - 2) * 3) = G0t; // Phi_(3+i)(3+i)
//     }
//     Phi.block<3, 3>(0, dimP - dimTheta) = -G1t * dt; // Phi_15
//     Phi.block<3, 3>(3, dimP - dimTheta) = Phi25L; // Phi_25
//     Phi.block<3, 3>(6, dimP - dimTheta) = Phi35L; // Phi_35
//     Phi.block<3, 3>(3, dimP - dimTheta + 3) = -G1t * dt; // Phi_26
//     Phi.block<3, 3>(6, dimP - dimTheta + 3) = -G0t * G2 * dt2; // Phi_36
    
//     return Phi;
// }


// // Compute Discrete noise matrix
// Eigen::MatrixXd DiscreteNoiseMatrix(Eigen::MatrixXd& Phi, double dt) {
//     int dimX = 23;
//     int dimTheta = 6;
//     int dimP = 69;
//     Eigen::MatrixXd G = Eigen::MatrixXd::Identity(dimP, dimP);

//     // Continuous noise covariance 
//     Eigen::MatrixXd Qc = Eigen::MatrixXd::Zero(dimP, dimP); // Landmark noise terms will remain zero
//     Qc.block<3, 3>(0, 0) = 0.01 * 0.01 * Eigen::Matrix3d::Identity(); // Gyroscope noise terms
//     Qc.block<3, 3>(3, 3) = 0.1 * 0.1 * Eigen::Matrix3d::Identity(); // Accelerometer noise terms
//     // Landmark noise terms will remain zero
//     Qc.block<3, 3>(dimP - dimTheta, dimP - dimTheta) = 0.00001 * 0.00001 * Eigen::Matrix3d::Identity(); // Gyroscope bias noise terms
//     Qc.block<3, 3>(dimP - dimTheta + 3, dimP - dimTheta + 3) = 0.0001 * 0.0001 * Eigen::Matrix3d::Identity(); // Accelerometer bias noise terms

//     // Noise Covariance Discretization
//     Eigen::MatrixXd PhiG = Phi * G;
//     Eigen::MatrixXd Qd = PhiG * Qc * PhiG.transpose() * dt; // Approximated discretized noise matrix (TODO: compute analytical) Eq. 61
//     return Qd;
// }

template <typename S>
bool EKF<S>::Predict(const IMU& imu) {
    assert(imu.timestamp_ >= current_time_);

    double dt = imu.timestamp_ - current_time_;
    if (dt > (5 * options_.imu_dt_) || dt < 0) {
        // 时间间隔不对，可能是第一个IMU数据，没有历史信息
        LOG(INFO) << "skip this imu because dt_ = " << dt;
        current_time_ = imu.timestamp_;
        return false;
    }

    // state 递推
    VecT new_p = p_ + v_ * dt + 0.5 * (R_ * (imu.acce_ - ba_)) * dt * dt + 0.5 * g_ * dt * dt; // 3.41a
    VecT new_v = v_ + R_ * (imu.acce_ - ba_) * dt + g_ * dt; // 3.41b
    SO3 new_R = R_ * SO3::exp((imu.gyro_ - bg_) * dt); // 3.41c Right multiply because IMU data is in local frame

    R_ = new_R;
    v_ = new_v;
    p_ = new_p;
    // 其余状态维度不变

    // error state 递推
    // 计算运动过程雅可比矩阵 F，见(3.42 or 3.47)
    // F实际上是稀疏矩阵，也可以不用矩阵形式进行相乘而是写成散装形式(faster)，这里为了教学方便，使用矩阵形式
    double dt2 = dt * dt;
    Mat18T F = Mat18T::Identity();                                                 // 主对角线
    // F.template block<3, 3>(0, 3) = Mat3T::Identity() * dt;                         // p 对 v
    // F.template block<3, 3>(3, 6) = -R_.matrix() * SO3::hat(imu.acce_ - ba_) * dt;  // v对theta
    // F.template block<3, 3>(3, 12) = -R_.matrix() * dt;                             // v 对 ba
    // F.template block<3, 3>(3, 15) = Mat3T::Identity() * dt;                        // v 对 g
    // F.template block<3, 3>(6, 6) = SO3::exp(-(imu.gyro_ - bg_) * dt).matrix();     // theta 对 theta
    // F.template block<3, 3>(6, 9) = -Mat3T::Identity() * dt;                        // theta 对 bg
    F.template block<3, 3>(0, 3) = Mat3T::Identity() * dt;                         // p 对 v
    // F.template block<3, 3>(0, 6) = 0; // -0.5 * dt2 * SO3::hat(imu.acce_ - ba_);  // v对theta
    // F.template block<3, 3>(0, 12) = -0.5 * dt2 * R_.matrix();                             // v 对 ba
    // F.template block<3, 3>(0, 15) = -0.5 * dt2 * Mat3T::Identity();                        // v 对 g
    F.template block<3, 3>(3, 6) = -R_.matrix() * SO3::hat(imu.acce_-ba_) * dt;                             // v 对 ba
    F.template block<3, 3>(3, 12) = -R_.matrix() * dt;                             // v 对 ba
    F.template block<3, 3>(3, 15) = Mat3T::Identity() * dt;                        // v 对 g
    F.template block<3, 3>(6, 6) = SO3::exp(-(imu.gyro_ - bg_) * dt).matrix();     // theta 对 theta
    F.template block<3, 3>(6, 9) = -Mat3T::Identity() * dt;                        // theta 对 bg

    // mean and cov prediction
    // dx_ = F * dx_;  // 这行其实没必要算，dx_在重置之后应该为零，因此这步可以跳过或注释掉
    // F需要参与cov部分计算，所以保留
    cov_ = F * cov_.eval() * F.transpose() + Q_; // P_pred: Predicted Covariance (3.48b)
    current_time_ = imu.timestamp_;
    return true;
}

template <typename S>
bool EKF<S>::ObserveWheelSpeed(const Odom& odom) {
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

    // dx_ = K * (vel_world - v_); // 3.68

    // update cov
    cov_ = (Mat18T::Identity() - K * H) * cov_;

    // UpdateAndReset();
    return true;
}

template <typename S>
bool EKF<S>::ObserveGps(const GNSS& gnss) {
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
bool EKF<S>::ObserveMoCap(const MoCap& mocap) {
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
bool EKF<S>::ObserveSE3(const SE3& pose, double trans_noise, double ang_noise) {
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
    // dx_ = K * innov; // 3.51b
    cov_ = (Mat18T::Identity() - K * H) * cov_;  // Corrected covariance (3.51d)

    this->UpdateAndReset(); // Apply 3.51c
    return true;
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

Eigen::MatrixXd Adjoint_SEK3(const Eigen::MatrixXd& X) {
    // Compute Adjoint(X) for X in SE_K(3)
    int K = X.cols() - 3;
    Eigen::MatrixXd Adj = Eigen::MatrixXd::Zero(3 + 3 * K, 3 + 3 * K);
    Eigen::Matrix3d R = X.block<3, 3>(0, 0);
    Adj.block<3, 3>(0, 0) = R;
    for (int i = 0; i < K; ++i) {
        Adj.block<3, 3>(3 + 3 * i, 3 + 3 * i) = R;
        Adj.block<3, 3>(3 + 3 * i, 0) = skewSymmetric(X.block<3, 1>(0, 3 + i)) * R;
    }
    return Adj;
}

void save_Pose_asTUM(std::string filename, SO3 orient, Vec3d tran, double t)
{
    std::ofstream save_points;
    save_points.setf(std::ios::fixed, std::ios::floatfield);
    save_points.open(filename.c_str(), std::ios::app);

    Eigen::Quaterniond q(orient.matrix());

    save_points.precision(9);
    save_points << t << " ";
    save_points.precision(10);
    save_points << tran(0) << " "
        << tran(1) << " "
        << tran(2) << " "
        << q.x() << " "
        << q.y() << " "
        << q.z() << " "
        << q.w() << std::endl;
}

template <typename S>
bool EKF<S>::ObserveLandmarks(const sad::Landmarks& landmarks) {
    // static std::vector<Vec3d> global_landmarks({ {0, 0, 0}, {0, 0, 6.5}, {10, 0, 0}, {10, 0, 6.5}, {10, 10, 0}, {10, 10, 6.5}, {0, 10, 0}, {0, 10, 6.5}, {0, 5, 10}, {10, 5, 10}, {0, 6, 0}, {0, 8, 0}, {0, 8, 5}, {0, 6, 5}, {0, 2, 2.5}, {0, 4, 2.5}, {0, 4, 5}, {0, 2, 5} });
    static std::vector<Vec3d> global_landmarks({ {0, 0, 6.5}, {10, 0, 0}, {10, 0, 6.5}, {10, 10, 0} });
    int numLandmarks = landmarks.landmarks_.size();
    
    // Resize observation matrix and observations vector to accommodate all landmarks
    // Observation matrix H is 54 * 69
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3 * numLandmarks, 18);
    Eigen::VectorXd observations = Eigen::VectorXd::Zero(3 * numLandmarks);
    Eigen::VectorXd measurements = Eigen::VectorXd::Zero(3 * numLandmarks);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(3 * numLandmarks, 3 * numLandmarks) * 0.1; // Observation noise, tune as necessary

    for (int i = 0; i < numLandmarks; ++i) {
        const Vec3d& landmark = landmarks.landmarks_[i].tail<3>();
        const Vec3d& landmark_local = R_.matrix().transpose() * (global_landmarks[i] - p_);

        // Set the observation vector
        observations.segment<3>(3 * i) = landmark;
        measurements.segment<3>(3 * i) = landmark_local; // landmarks.landmarks_[i].tail<3>();
        H.block<3, 3>(3 * i, 0) = -R_.matrix().transpose(); // Partial derivative wrt position
        H.block<3, 3>(3 * i, 6) = skewSymmetric(R_.matrix().transpose() * (global_landmarks[i] - p_)); // Partial derivative wrt orientation
    }

    Eigen::MatrixXd SS = H * cov_ * H.transpose() + R; // 54*54 = 54*18 * 18*18 * 18*54 + 54*54

    // Calculate the Kalman gain
    Eigen::MatrixXd K = cov_ * H.transpose() * SS.inverse();         // 18 * 54

    // Calculate the innovation (measurement residual)
    Eigen::VectorXd innovation = observations - measurements;   // 54 * 1

    Eigen::VectorXd dx = K * innovation; // 3.51b  // 18 * 1

    // Do not update the bias and gravity terms
    p_ += dx.template block<3, 1>(0, 0);
    v_ += dx.template block<3, 1>(3, 0);
    R_ = R_ * SO3::exp(dx.template block<3, 1>(6, 0));
    cov_ = (Mat18T::Identity() - K * H) * cov_;  // Corrected covariance (3.51d)

    save_Pose_asTUM("log/pose_ekf.txt", R_, p_, landmarks.timestamp_);

    // this->UpdateAndReset(); // Apply 3.51c

    return true;
}

}  // namespace sad

#endif  // SLAM_IN_AUTO_DRIVING_EKF_HPP
