//
// Created by Shifan on 2024/07/08.
//

#include "ch3/iekf.hpp"
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
    sad::IEKFD iekf;

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

    std::ofstream fout("./data/ch3/landmarks_corrected_result.txt");
    bool imu_inited = false, gnss_inited = false;

    std::shared_ptr<sad::ui::PangolinWindow> ui = nullptr;
    if (FLAGS_with_ui) {
        ui = std::make_shared<sad::ui::PangolinWindow>();
        ui->Init();
    }

    /// 设置各类回调函数
    bool first_gnss_set = false;
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
              // 读取初始零偏，设置IEKF
              sad::IEKFD::Options options;
              // 噪声由静止初始化器估计
              options.gyro_var_ = sqrt(imu_init.GetCovGyro()[0]);
              options.acce_var_ = sqrt(imu_init.GetCovAcce()[0]);
              LOG(INFO) << "imu_init.GetGravity() " << imu_init.GetGravity().transpose();
              iekf.SetInitialConditions(options, imu_init.GetInitBg(), imu_init.GetInitBa(), imu_init.GetGravity());
              imu_inited = true;
              return;
          }

          iekf.Predict(imu);

          /// predict就会更新IEKF，所以此时就可以发送数据
          auto state = iekf.GetNominalState();
          if (ui) {
              ui->UpdateNavState(state);

            //   Eigen::Matrix3d R;   // 把body坐标系朝向旋转一下,得到相机坐标系，好让它看到landmark,  相机坐标系的轴在body坐标系中的表示
            //   // 相机朝着轨迹里面看， 特征点在轨迹外部， 这里我们采用这个
            //   R << 0, 0, -1,
            //       -1, 0, 0,
            //       0, 1, 0;
            //   SO3 R_bc(R);
            //   Eigen::Vector3d t_bc = Eigen::Vector3d(0.05, 0.04, 0.03);

            //   sad::NavStated body_steate = sad::NavStated(state.timestamp_, state.R_ * R_bc, state.p_ + state.R_ * t_bc, state.v_, state.bg_, state.ba_);
            //   ui->UpdateNavState(body_steate);
          }

          /// 记录数据以供绘图
          save_result(fout, state);

          usleep(1e3);
      })
    .SetLandmarksProcessFunc([&](const sad::Landmarks& landmarks) {
        if (!imu_inited) {
            return;
        }
        /// Landmarks 处理函数
        // LOG(INFO) << "time = " << landmarks.timestamp_;
        // // print features
        // for (int i = 0; i < landmarks.landmarks_.size(); i++) {
        //     LOG(INFO) << "feature " << i << ": " << landmarks.timestamp_ << " " << landmarks.landmarks_[i].transpose();
        // }

        // Eigen::Matrix3d R_bc;   // 把body坐标系朝向旋转一下,得到相机坐标系，好让它看到landmark,  相机坐标系的轴在body坐标系中的表示
        // // 相机朝着轨迹里面看， 特征点在轨迹外部， 这里我们采用这个
        // R_bc << 0, 0, -1,
        //     -1, 0, 0,
        //     0, 1, 0;
        // Eigen::Vector3d t_bc = Eigen::Vector3d(0.05, 0.04, 0.03);

        // // return NavStated(timestamp_, R_ * R_bc, p_ + R_ * t_bc, v_, bg_, ba_);

        // // convert landmarks to body frame
        // sad::Landmarks landmarks_body;
        // landmarks_body.timestamp_ = landmarks.timestamp_;
        // for (int i = 0; i < landmarks.landmarks_.size(); i++) {
        //     Vec3d landmark_c = R_bc* landmarks.landmarks_[i].tail<3>() + t_bc;
        //     landmarks_body.landmarks_.push_back(Vec4d(landmarks.landmarks_[i][0], landmark_c[0], landmark_c[1], landmark_c[2])); // Rbc
        // }

        // iekf.ObserveLandmarks(landmarks);
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