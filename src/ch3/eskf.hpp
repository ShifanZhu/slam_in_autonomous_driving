//
// Created by xiang on 2021/11/11.
//

#ifndef SLAM_IN_AUTO_DRIVING_ESKF_HPP
#define SLAM_IN_AUTO_DRIVING_ESKF_HPP

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
 * 本书使用18维的ESKF，标量类型可以由S指定，默认取double
 * 变量顺序：p, v, R, bg, ba, grav，与书本对应
 * @tparam S    状态变量的精度，取float或double
 */
template <typename S = double>
class ESKF {
   public:
    /// 类型定义
    using SO3 = Sophus::SO3<S>;                     // 旋转变量类型
    using VecT = Eigen::Matrix<S, 3, 1>;            // 向量类型
    using Vec18T = Eigen::Matrix<S, 18, 1>;         // 18维向量类型
    using Mat3T = Eigen::Matrix<S, 3, 3>;           // 3x3矩阵类型
    using MotionNoiseT = Eigen::Matrix<S, 18, 18>;  // 运动噪声类型
    using OdomNoiseT = Eigen::Matrix<S, 3, 3>;      // 里程计噪声类型
    using GnssNoiseT = Eigen::Matrix<S, 6, 6>;      // GNSS噪声类型
    using Mat18T = Eigen::Matrix<S, 18, 18>;        // 18维方差类型
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
    ESKF(Options option = Options()) : options_(option) { BuildNoise(option); }

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
    void SetCov(const Mat18T& cov) { cov_ = cov; }

    /// 获取重力
    Vec3d GetGravity() const { return g_; }
    SO3 GetOrientation() const { return R_; }
    void SetOrientation(SO3& R) { R_ = R; }

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
        Q_.diagonal() << 0, 0, 0, ev2, ev2, ev2, et2, et2, et2, eg2, eg2, eg2, ea2, ea2, ea2, 0, 0, 0;

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
            LOG(INFO) << "update bg: " << bg_.transpose();
        }

        if (options_.update_bias_acce_) {
            ba_ += dx_.template block<3, 1>(12, 0);
            LOG(INFO) << "update ba: " << ba_.transpose();
        }

        // todo
        // g_ += dx_.template block<3, 1>(15, 0); //? why we update gravity since its derivetive is zero 3.25f?
        //// LOG(INFO) << "update delta g: " << dx_.template block<3, 1>(15, 0).transpose();

        ProjectCov(); // 3.63
        dx_.setZero(); //? why we set dx to zero?
    }

    /// 对P阵进行投影，参考式(3.63)
    void ProjectCov() {
        Mat18T J = Mat18T::Identity();
        J.template block<3, 3>(6, 6) = Mat3T::Identity() - 0.5 * SO3::hat(dx_.template block<3, 1>(6, 0)); // 3.61
        // LOG(INFO) << "project affect: " << (0.5 * SO3::hat(dx_.template block<3, 1>(6, 0)));
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
    Vec18T dx_ = Vec18T::Zero();

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

using ESKFD = ESKF<double>;
using ESKFF = ESKF<float>;


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
bool ESKF<S>::Predict(const IMU& imu) {
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

    R_ = new_R;
    v_ = new_v;
    p_ = new_p;
    // 其余状态维度不变

    // error state 递推
    // 计算运动过程雅可比矩阵 F，见(3.42 or 3.47)
    // F实际上是稀疏矩阵，也可以不用矩阵形式进行相乘而是写成散装形式(faster)，这里为了教学方便，使用矩阵形式
    Mat18T F = Mat18T::Identity();                                                 // 主对角线
    F.template block<3, 3>(0, 3) = Mat3T::Identity() * dt;                         // p 对 v
    F.template block<3, 3>(3, 6) = -R_.matrix() * SO3::hat(imu.acce_ - ba_) * dt;  // v对theta
    F.template block<3, 3>(3, 12) = -R_.matrix() * dt;                             // v 对 ba
    F.template block<3, 3>(3, 15) = Mat3T::Identity() * dt;                        // v 对 g
    F.template block<3, 3>(6, 6) = SO3::exp(-(imu.gyro_ - bg_) * dt).matrix();     // theta 对 theta
    F.template block<3, 3>(6, 9) = -Mat3T::Identity() * dt;                        // theta 对 bg

    // mean and cov prediction
    // LOG(INFO) << "dx1 = " << dx_.transpose();
    // dx_ = F * dx_;  // 这行其实没必要算，dx_在重置之后应该为零，因此这步可以跳过或注释掉
    // LOG(INFO) << "dx2 = " << dx_.transpose();
    // LOG(INFO)<<"Q_ = "<<Q_.diagonal().transpose();
    // F需要参与cov部分计算，所以保留
    cov_ = F * cov_.eval() * F.transpose() + Q_; // P_pred: Predicted Covariance (3.48b)
    current_time_ = imu.timestamp_;
    return true;
}

template <typename S>
bool ESKF<S>::ObserveWheelSpeed(const Odom& odom) {
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
    cov_ = (Mat18T::Identity() - K * H) * cov_;

    UpdateAndReset();
    return true;
}

template <typename S>
bool ESKF<S>::ObserveGps(const GNSS& gnss) {
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
bool ESKF<S>::ObserveMoCap(const MoCap& mocap) {
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
bool ESKF<S>::ObserveSE3(const SE3& pose, double trans_noise, double ang_noise) {
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
    cov_ = (Mat18T::Identity() - K * H) * cov_;  // Corrected covariance (3.51d)

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

template <typename S>
bool ESKF<S>:: ObserveLandmarks(const sad::Landmarks& landmarks) {
    // static std::vector<Vec3d> global_landmarks({ {0, 0, 0}, {0, 0, 6.5}, {10, 0, 0}, {10, 0, 6.5}, {10, 10, 0}, {10, 10, 6.5}, {0, 10, 0}, {0, 10, 6.5}, {0, 5, 10}, {10, 5, 10}, {0, 6, 0}, {0, 8, 0}, {0, 8, 5}, {0, 6, 5}, {0, 2, 2.5}, {0, 4, 2.5}, {0, 4, 5}, {0, 2, 5} });
    static std::vector<Vec3d> global_landmarks({ {0, 0, 6.5}, {10, 0, 0}, {10, 0, 6.5}, {10, 10, 0} });
    int numLandmarks = landmarks.landmarks_.size();
    
    // Resize observation matrix and observations vector to accommodate all landmarks
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3 * numLandmarks, 18);
    Eigen::MatrixXd x_dx = Eigen::MatrixXd::Identity(18, 18);
    Eigen::VectorXd observations = Eigen::VectorXd::Zero(3 * numLandmarks);
    Eigen::VectorXd measurements = Eigen::VectorXd::Zero(3 * numLandmarks);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(3 * numLandmarks, 3 * numLandmarks) * 0.1; // Observation noise, tune as necessary
    for (int i = 0; i < numLandmarks; ++i) {
        const Vec3d& landmark = landmarks.landmarks_[i].tail<3>();
        const Vec3d& landmark_local = R_.matrix().transpose() * (global_landmarks[i] - p_);

        // Set the observation vector
        observations.segment<3>(3 * i) = landmark;
        measurements.segment<3>(3 * i) = landmark_local; // landmarks.landmarks_[i].tail<3>();
        // std::cout << "time: " << current_time_ << std::endl;
        // std::cout << "observation: " << landmark.transpose() << std::endl;
        // std::cout << "measurement: " << measurements.segment<3>(3 * i).transpose() << std::endl;
        // std::cout << "R_: " << R_.matrix() << std::endl;
        // std::cout << "p_: " << p_.transpose() << std::endl;

        // auto diff = (landmark - measurements.segment<3>(3 * i)).transpose();
        // if (diff[0] > 0.5) {
        //     std::cout << "=========================" << std::endl;
        //     std::cout << "diff: " << diff << std::endl;
        //     std::cout << "time: " << current_time_ << std::endl;
        //     std::cout << "t: " << p_.transpose() << std::endl;
        //     std::cout << "R in quaternion: " << R_.unit_quaternion().coeffs().transpose() << " " << R_.unit_quaternion().coeffs().transpose().w() << std::endl;
        //     std::cout << "global_landmark:" << global_landmarks[i].transpose() << std::endl;
        //     std::cout << "landmark:" << landmarks.landmarks_[i].transpose() << std::endl;
        //     std::cout << "landmark_local:" << landmark_local.transpose() << std::endl;
        // }
        // Compute the Jacobian for the current landmark
        H.block<3, 3>(3 * i, 0) = -R_.matrix().transpose(); // Partial derivative wrt position
        // H.block<3, 3>(3 * i, 3) = Eigen::Matrix3d::Zero(); // Partial derivative wrt velocity
        H.block<3, 3>(3 * i, 6) = skewSymmetric(R_.matrix().transpose() * (global_landmarks[i] - p_)); // Partial derivative wrt orientation
        // H.block<3, 3>(3 * i, 9) = Eigen::Matrix3d::Zero(); // Partial derivative wrt gyroscope bias
        // H.block<3, 3>(3 * i, 12) = Eigen::Matrix3d::Zero(); // Partial derivative wrt accelerometer bias
        // H.block<3, 3>(3 * i, 15) = Eigen::Matrix3d::Zero(); // Partial derivative wrt gravity vector
    }

    Eigen::Vector3d dq = dx_.segment(6, 3) * 0.5;
    x_dx.block<3, 3>(6, 6) = SO3::jr(dq).inverse();
    // LOG(INFO)<<"x_dx: "<<x_dx;
    H = H * x_dx;

    // LOG(INFO) << "H dim: " << H.rows() << "x" << H.cols();      // 54 * 18
    // LOG(INFO) << "observations dim: " << observations.rows() << "x" << observations.cols();                     // 54 * 1
    // LOG(INFO) << "measurements dim: " << measurements.rows() << "x" << measurements.cols();                     // 54 * 1
    // LOG(INFO) << "cov_ dim: " << cov_.rows() << "x" << cov_.cols();                                             // 18 * 18
    // LOG(INFO) << "R dim: " << R.rows() << "x" << R.cols();         // 54 * 54
    // Calculate the innovation covariance
    Eigen::MatrixXd SS = H * cov_ * H.transpose() + R;

    // std::cout << "cov_ = " << std::endl << cov_ << std::endl;
    // std::cout << "R = " << std::endl << R << std::endl;

    // Calculate the Kalman gain
    Eigen::MatrixXd K = cov_ * H.transpose() * SS.inverse();         // 18 * 54

    // Calculate the innovation (measurement residual)
    Eigen::VectorXd innovation = observations - measurements;   // 54 * 1
    // std::cout << "innovation: " << innovation.transpose() << std::endl;

    // Update the error state estimate
    dx_ = K * innovation; // 3.51b  // 18 * 1
    // std::cout << "dx_: " << dx_.transpose() << std::endl;
    // std::cout << "innovation: " << innovation.transpose() << std::endl;
    cov_ = (Mat18T::Identity() - K * H) * cov_;  // Corrected covariance (3.51d)

    this->UpdateAndReset(); // Apply 3.51c


    save_Pose_asTUM("log/pose_eskf.txt", R_, p_, landmarks.timestamp_);


    return true;
}

}  // namespace sad

#endif  // SLAM_IN_AUTO_DRIVING_ESKF_HPP
