//
// Created by Shifan on 2024/07/08.
//

#include "ch3/ekf.hpp"
#include "ch3/static_imu_init.h"
#include "common/io_utils.h"
#include "tools/ui/pangolin_window.h"
#include "utm_convert.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <fstream>
#include <iomanip>

DEFINE_string(txt_path, "./data/ch3/imu_points_imu_sim.txt", "数据文件路径");

DEFINE_bool(with_ui, true, "是否显示图形界面");

/**
 * 本程序演示使用3D Landmark+IMU进行组合导航
 */
int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);

    if (fLS::FLAGS_txt_path.empty()) {
        return -1;
    }

    // 初始化器
    sad::StaticIMUInit imu_init;  // 使用默认配置
    sad::EKFD ekf;

    sad::TxtIO io(FLAGS_txt_path);

    auto save_vec3 = [](std::ofstream& fout, const Vec3d& v) { fout << v[0] << " " << v[1] << " " << v[2] << " "; };
    auto save_quat = [](std::ofstream& fout, const Quatd& q) {
        fout << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " ";
    };

    auto save_result = [&save_vec3, &save_quat](std::ofstream& fout, const sad::NavStated& save_state) {
        fout << std::setprecision(18) << save_state.timestamp_ << " " << std::setprecision(9);
        save_vec3(fout, save_state.p_);
        save_quat(fout, save_state.R_.unit_quaternion());
        save_vec3(fout, save_state.v_);
        save_vec3(fout, save_state.bg_);
        save_vec3(fout, save_state.ba_);
        fout << std::endl;
    };

    std::ofstream fout("./data/ch3/landmarks_ekf_result.txt");
    bool imu_inited = false, gnss_inited = false;

    std::shared_ptr<sad::ui::PangolinWindow> ui = nullptr;
    if (FLAGS_with_ui) {
        ui = std::make_shared<sad::ui::PangolinWindow>();
        ui->Init();
    }

    /// 设置各类回调函数
    bool first_gnss_set = false;
    double prev_disturb_time = 0;

    Vec3d origin = Vec3d::Zero();

    // Set all these process callback functions in IO.
    io.SetIMUProcessFunc([&](const sad::IMU& imu) {
          /// IMU 处理函数. Static initilization first
          if (!imu_init.InitSuccess()) {
              imu_init.AddIMU(imu);
              return;
          }

          /// 需要IMU初始化
          if (!imu_inited) {
              // 读取初始零偏，设置EKF
              sad::EKFD::Options options;
              // 噪声由静止初始化器估计
              // comment out to use defaule value
            //   options.gyro_var_ = sqrt(imu_init.GetCovGyro()[0]);
            //   options.acce_var_ = sqrt(imu_init.GetCovAcce()[0]);
              LOG(INFO) << "imu_init.GetGravity() " << imu_init.GetGravity().transpose();
              ekf.SetInitialConditions(options, imu_init.GetInitBg(), imu_init.GetInitBa(), imu_init.GetGravity());
              imu_inited = true;
              return;
          }

          ekf.Predict(imu);

          /// predict就会更新ekf，所以此时就可以发送数据
          auto state = ekf.GetNominalState();
          if (ui) {
              ui->UpdateNavState(state);
          }

          /// 记录数据以供绘图
          save_result(fout, state);

          usleep(1e3);
      })
    .SetLandmarksProcessFunc([&](const sad::Landmarks& landmarks) {
        if (!imu_inited) {
            return;
        }
        // if (abs(landmarks.timestamp_ - 20) < 0.001) {
        if (abs(landmarks.timestamp_ - prev_disturb_time) > 2.999) {
            prev_disturb_time = landmarks.timestamp_;
            LOG(INFO) << "landmarks.timestamp_ = " << landmarks.timestamp_;
            double roll = 0;  // phi
            double pitch = 0; // theta
            double yaw = 30;   // psi
            Eigen::Matrix3d R_roll = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()).toRotationMatrix();
            Eigen::Matrix3d R_pitch = Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()).toRotationMatrix();
            Eigen::Matrix3d R_yaw = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
            Eigen::Matrix3d Rwb = R_yaw * R_pitch * R_roll;
            // get roll, pitch, yaw
            Vec3d euler_before = ekf.GetOrientation().matrix().eulerAngles(2, 1, 0);
            LOG(INFO) << "euler_before = " << euler_before.transpose() * 180 / M_PI;

            SO3 R_biased = ekf.GetOrientation() * SO3(Rwb);
            ekf.SetOrientation(R_biased);
            Vec3d euler_after = ekf.GetOrientation().matrix().eulerAngles(2, 1, 0);
            LOG(INFO) << "euler_after = " << euler_after.transpose() * 180 / M_PI;
            // R_ = SO3::exp(VecT(1.5, -1.6, 1.9));
        }
        /// Landmarks 处理函数
        ekf.ObserveLandmarks(landmarks);
    })
    .Go();

    while (ui && !ui->ShouldQuit()) {
        usleep(1e5);
    }
    if (ui) {
        ui->Quit();
    }
    return 0;
}