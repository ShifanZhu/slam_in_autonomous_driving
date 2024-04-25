//
// Created by Shifan on 2024/04/19.
//

#include "ch3/eskf.hpp"
#include "ch3/static_imu_init.h"
#include "common/io_utils.h"
#include "tools/ui/pangolin_window.h"
#include "utm_convert.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <fstream>
#include <iomanip>

// DEFINE_string(txt_path, "./data/ch3/eats_in_imu_imu.txt", "数据文件路径");
// DEFINE_string(txt_path, "./data/ch3/eats_imu.txt", "数据文件路径");
// DEFINE_string(txt_path, "./data/ch3/mocap1.txt", "数据文件路径");
DEFINE_string(txt_path, "./data/ch3/fasterlio_imu.txt", "数据文件路径");
DEFINE_bool(with_ui, true, "是否显示图形界面");

/**
 * 本程序演示使用MoCap+IMU进行组合导航
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
    sad::ESKFD eskf;

    sad::TxtIO io(FLAGS_txt_path);

    auto save_vec3 = [](std::ofstream& fout, const Vec3d& v) { fout << v[0] << " " << v[1] << " " << v[2] << " "; };
    auto save_quat = [](std::ofstream& fout, const Quatd& q) {
        fout << q.w() << " " << q.x() << " " << q.y() << " " << q.z();
    };

    auto save_result = [&save_vec3, &save_quat](std::ofstream& fout, const sad::NavStated& save_state) {
        fout << std::setprecision(18) << save_state.timestamp_ << " " << std::setprecision(9);
        save_vec3(fout, save_state.p_);
        save_quat(fout, save_state.R_.unit_quaternion());
        // save_vec3(fout, save_state.v_);
        // save_vec3(fout, save_state.bg_);
        // save_vec3(fout, save_state.ba_);
        fout << std::endl;
    };

    std::ofstream fout("./data/ch3/mocap1_result.txt");
    bool imu_inited = false, mocap_inited = false;

    std::shared_ptr<sad::ui::PangolinWindow> ui = nullptr;
    if (FLAGS_with_ui) {
        ui = std::make_shared<sad::ui::PangolinWindow>();
        ui->Init();
    }

    /// 设置各类回调函数
    bool first_mocap_set = false;
    Vec3d origin = Vec3d::Zero();

    // Set all these process callback functions in IO.
    io.SetIMUProcessFunc([&](const sad::IMU& imu) {
        // Eigen::Matrix<double, 4, 4> mi_matrix;
        // mi_matrix <<
        //     0.0355956346904962, -0.999212371176487, -0.0175381891551868, -0.163444825804024,
        //     -0.00894894931538293, -0.0178673034275079, 0.99980031795073, -0.0388815779414504,
        //     -0.999326206549759, -0.0354315785153829, -0.00957789887067184, -0.0504714801204347,
        //     0, 0, 0, 1;
        // SE3 T_mi(mi_matrix);
        // sad::IMU imu = imu_tmp;
        // imu.acce_ = T_mi.rotationMatrix() * imu.acce_;
        // imu.gyro_ = T_mi.rotationMatrix() * imu.gyro_;
        /// IMU 处理函数. Static initilization first
          if (!imu_init.InitSuccess()) {
              imu_init.AddIMU(imu);
              return;
          }

          /// 需要IMU初始化
          if (!imu_inited) {
              // 读取初始零偏，设置ESKF
              sad::ESKFD::Options options;
              // 噪声由静止初始化器估计
              options.gyro_var_ = sqrt(imu_init.GetCovGyro()[0]);
              options.acce_var_ = sqrt(imu_init.GetCovAcce()[0]);
              eskf.SetInitialConditions(options, imu_init.GetInitBg(), imu_init.GetInitBa(), imu_init.GetGravity());
              imu_inited = true;
              return;
          }

          if (!mocap_inited) {
              /// 等待有效的MoCap数据
              return;
          }

          /// MoCap 也接收到之后，再开始进行预测
          eskf.Predict(imu);

          /// predict就会更新ESKF，所以此时就可以发送数据
          auto state = eskf.GetNominalState();
          if (ui) {
              ui->UpdateNavState(state);
          }

          /// 记录数据以供绘图
          save_result(fout, state);

          usleep(1e3);
      })
        .SetMoCapProcessFunc([&](const sad::MoCap& mocap) {
            /// MoCap 处理函数
            if (!imu_inited) {
                return;
            }
            sad::MoCap mocap_convert = mocap;
            double angleX = M_PI / 2 + 1*M_PI/180;
            double angleZ = -M_PI / 2;
            Eigen::AngleAxisd rollAngle(angleX, Eigen::Vector3d::UnitX());
            Eigen::AngleAxisd yawAngle(angleZ, Eigen::Vector3d::UnitZ());
            Eigen::Matrix3d rotationMatrix = (rollAngle * yawAngle).toRotationMatrix();

            Eigen::Matrix<double, 4, 4> T_marker_imu_matrix;
            Eigen::Matrix<double, 4, 4> T_lidar_imu_matrix;
            Eigen::Matrix<double, 4, 4> T_rgb_imu_matrix;
            T_marker_imu_matrix <<
                0.0355956346904962, -0.999212371176487, -0.0175381891551868, 0.043163444825804024,
                -0.00894894931538293, -0.0178673034275079, 0.99980031795073, -0.013188815779414504,
                -0.999326206549759, -0.0354315785153829, -0.00957789887067184, -0.00504714801204347,
                0, 0, 0, 1;
            T_marker_imu_matrix.block<3, 3>(0, 0) = rotationMatrix;
            T_lidar_imu_matrix << 0.999777438410932, 0.0210687633451775, 0.00108667179048293, 0.0619280375186583,
                0.0210682887473136, -0.999777939330872, 0.000446359245352153, -0.00274487391903187,
                0.00109583472072527, -0.000423365587873757, -0.999999309953684, -0.0784247982183386,
                0, 0, 0, 1;
            T_rgb_imu_matrix << -0.00357268534012919, 0.999978105031021, 0.00557004291095844, -0.016642751623471,
                0.00242922070831184, -0.00556138326250457, 0.999981584781919, 0.0131737658918879,
                0.999990667359557, 0.00358615041193459, -0.00240929844814716, -0.00632544533652158,
                -0, 0, -0, 1;
            SE3 mocap_pose(mocap.GetSE3());
            static SE3 first_mocap_pose = mocap_pose;
            // LOG(INFO) << "mocap_pose " << mocap.GetSE3().matrix();
            SE3 T_marker_imu(T_marker_imu_matrix);
            SE3 T_lidar_imu(T_lidar_imu_matrix);
            SE3 T_rgb_imu(T_rgb_imu_matrix);
            SE3 T_mocap_imu(T_marker_imu.inverse() * first_mocap_pose.inverse() * mocap_pose* T_marker_imu);
            SE3 T_lidar0_imu(mocap_pose * T_lidar_imu);
            SE3 T_imu0_imu(T_rgb_imu.inverse() * mocap_pose* T_rgb_imu);
            // mocap_convert.SetSE3(T_lidar0_imu);
            // LOG(INFO) << "mocap_convert " << mocap_convert.GetSE3().matrix();
            // LOG(INFO) << "T_marker_imu euler angle : " << T_marker_imu.rotationMatrix().eulerAngles(0, 1, 2).transpose() * 180 / 3.14159; // roll pitch yaw
            // LOG(INFO) << "T_lidar_imu euler angle : " << T_lidar_imu.rotationMatrix().eulerAngles(0, 1, 2).transpose() * 180 / 3.14159; // roll pitch yaw
            // LOG(INFO) << "T_rgb_imu euler angle : " << T_rgb_imu.rotationMatrix().eulerAngles(1, 2, 0).transpose() * 180 / 3.14159; // roll pitch yaw

            // mocap_convert.position_ = T_rgb0_imu.translation();
            // mocap_convert.orientation_ = T_rgb0_imu.so3().unit_quaternion();

            // // mocap_convert stores the converted GNSS data for following use.
            // sad::GNSS gnss_convert = gnss;
            // if (!sad::ConvertGps2UTM(gnss_convert, antenna_pos, FLAGS_antenna_angle) || !gnss_convert.heading_valid_) {
            //     return;
            // }

            /// 得到第一帧的原点，并去掉原点
            if (!first_mocap_set) {
                // origin = gnss_convert.utm_pose_.translation();
                origin = mocap_convert.position_;
                first_mocap_set = true;
            }
            mocap_convert.position_ -= origin;

            // 要求RTK heading有效，才能合入ESKF
            eskf.ObserveMoCap(mocap_convert);

            auto state = eskf.GetNominalState();
            if (ui) {
                ui->UpdateNavState(state);
            }
            save_result(fout, state);

            mocap_inited = true;
        })
        // .SetOdomProcessFunc([&](const sad::Odom& odom) {
        //     /// Odom 处理函数，本章Odom只给初始化使用
        //     imu_init.AddOdom(odom);
        //     if (FLAGS_with_odom && imu_inited && mocap_inited) {
        //         eskf.ObserveWheelSpeed(odom);
        //     }
        // })
        .Go();

    while (ui && !ui->ShouldQuit()) {
        usleep(1e5);
    }
    if (ui) {
        ui->Quit();
    }
    return 0;
}