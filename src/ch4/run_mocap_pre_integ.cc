//
// Created by Shifan on 2024/04/25.
//

#include "ch3/static_imu_init.h"
#include "ch3/utm_convert.h"
#include "ch4/gins_pre_integ.h"
#include "common/io_utils.h"
#include "tools/ui/pangolin_window.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <fstream>
#include <iomanip>

/**
 * 运行由预积分构成的GINS系统
 */
// DEFINE_string(txt_path, "./data/ch3/mocap1.txt", "数据文件路径");
// DEFINE_string(txt_path, "./data/ch3/fasterlio_imu.txt", "数据文件路径");
// DEFINE_string(txt_path, "./data/ch3/mh01.txt", "数据文件路径");
// DEFINE_string(txt_path, "./data/ch3/eats_in_imu_imu.txt", "数据文件路径");
// DEFINE_string(txt_path, "./data/ch3/vector/corridors.txt", "数据文件路径");
// DEFINE_string(txt_path, "./data/ch3/eagle/eats_imu.txt", "数据文件路径");
// DEFINE_string(txt_path, "./data/ch3/eagle/gt_imu.txt", "数据文件路径");
DEFINE_string(txt_path, "./data/ch3/eagle/outdoor/sidewalk1_day_trot/lidar_imu.txt", "数据文件路径");
// DEFINE_string(txt_path, "./data/ch3/eagle/outdoor/sidewalk1_day_trot/eats_imu.txt", "数据文件路径");
// DEFINE_string(txt_path, "./data/ch3/eagle/indoor/mocap_env3_trot/mocap_imu.txt", "数据文件路径");
// DEFINE_string(txt_path, "./data/ch3/eagle/indoor/mocap_env3_trot/eats_imu.txt", "数据文件路径");
DEFINE_bool(with_ui, true, "是否显示图形界面");
DEFINE_bool(debug, false, "是否打印调试信息");


Eigen::Matrix3d correctRotationMatrix(const Eigen::Matrix3d& R) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R_corrected = svd.matrixU() * svd.matrixV().transpose();
    return R_corrected;
}

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

    sad::TxtIO io(fLS::FLAGS_txt_path);

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

    std::ofstream fout("./data/ch3/eagle/mocap_preintg.txt");
    bool imu_inited = false, mocap_inited = false;

    sad::GinsPreInteg::Options gins_options;
    gins_options.verbose_ = FLAGS_debug;
    sad::GinsPreInteg gins(gins_options);

    bool first_mocap_set = false;
    Vec3d origin = Vec3d::Zero();

    std::shared_ptr<sad::ui::PangolinWindow> ui = nullptr;
    if (FLAGS_with_ui) {
        ui = std::make_shared<sad::ui::PangolinWindow>();
        ui->Init();
    }

    /// 设置各类回调函数
    io.SetIMUProcessFunc([&](const sad::IMU& imu) {
          /// IMU 处理函数, static initialization
          if (!imu_init.InitSuccess()) {
              imu_init.AddIMU(imu);
              return;
          }

          /// 需要IMU初始化
          if (!imu_inited) {
              // 读取初始零偏，设置GINS
              sad::GinsPreInteg::Options options;
              options.preinteg_options_.init_bg_ = imu_init.GetInitBg();
              options.preinteg_options_.init_ba_ = imu_init.GetInitBa();
              options.gravity_ = imu_init.GetGravity();
              gins.SetOptions(options);
              imu_inited = true;
              return;
          }

          if (!mocap_inited) {
              /// 等待有效的RTK数据
              return;
          }

          /// Only Integrate here, when MoCap 也接收到之后，再开始进行预测
          gins.AddImu(imu);

          auto state = gins.GetState();
          save_result(fout, state);
          if (ui) {
              ui->UpdateNavState(state);
              usleep(5e2);
          }
      })
        .SetMoCapProcessFunc([&](const sad::MoCap& mocap) {
            /// MoCap 处理函数
            // IMU should be inited first
            if (!imu_inited) {
                return;
            }

            sad::MoCap mocap_convert = mocap;
            // if (!sad::ConvertGps2UTM(gnss_convert, antenna_pos, FLAGS_antenna_angle) || !gnss_convert.heading_valid_) {
            //     return;
            // }

            Eigen::Matrix<double, 4, 4> T_marker_imu_matrix, T_rgb_imu_matrix;

            SE3 mocap_pose(mocap.GetSE3());
            T_marker_imu_matrix << 0.00676736861588201, 0.999787658357018, 0.0194638361842693, -0.017683758,
                -0.0109922982240422, 0.0195374824093171, -0.999748696503706, -0.0195426,
                -0.999916682579989, 0.00655171566056689, 0.0111221814466671, -0.00486914,
                0, 0, 0, 1;

            LOG(INFO)<<"quaternion of T_marker_imu: "<<std::setprecision(12)<<SE3(T_marker_imu_matrix).unit_quaternion().coeffs().transpose()<<std::endl;
            T_rgb_imu_matrix << -0.00357268534012919, 0.999978105031021, 0.00557004291095844, -0.016642751623471,
                0.00242922070831184, -0.00556138326250457, 0.999981584781919, 0.0131737658918879,
                0.999990667359557, 0.00358615041193459, -0.00240929844814716, -0.00632544533652158,
                -0, 0, -0, 1;
            // LOG(INFO) << "roll pitch yaw: " << T_rgb_imu_matrix.matrix().block<3, 3>(0, 0).eulerAngles(0, 1, 2).transpose() * 180 / M_PI << std::endl; // roll pitch yaw

            static SE3 first_mocap_pose = mocap_pose;
            SE3 T_marker_imu(T_marker_imu_matrix);
            SE3 T_mocap_imu(T_marker_imu.inverse() * first_mocap_pose.inverse() * mocap_pose * T_marker_imu);
            SE3 T_rgb_imu(T_rgb_imu_matrix);
            SE3 T_imu0_imu(T_rgb_imu.inverse() * first_mocap_pose.inverse() * mocap_pose* T_rgb_imu);

            // mocap_convert.SetSE3(T_imu0_imu);


            
            // Eigen::Matrix<double, 4, 4> T_adjust = Eigen::Matrix<double, 4, 4>::Identity();
            // double angleX = -2 * M_PI / 180;
            // double angleY = 3 * M_PI / 180;
            // double angleZ = -5 * M_PI / 180;
            // Eigen::AngleAxisd rollAngle(angleX, Eigen::Vector3d::UnitX());
            // Eigen::AngleAxisd pitchAngle(angleY, Eigen::Vector3d::UnitY());
            // Eigen::AngleAxisd yawAngle(angleZ, Eigen::Vector3d::UnitZ());
            // T_adjust.block<3, 3>(0, 0) = (rollAngle * pitchAngle * yawAngle).toRotationMatrix();
            // SE3 T_small_adjust_se3(T_adjust);
            // // // mocap_convert.SetSE3(SE3(T_adjust) * mocap.GetSE3());

            // // // Vector large scale dataset
            // Eigen::Matrix<double, 4, 4> T_camera_imu_matrix, T_lidar_camera_matrix, T_camera_lidar_matrix, T_lidar_imu_matrix;
            // T_camera_lidar_matrix << 0.0119197, -0.999929, 0.0000523, 0.0853154,
            //     -0.00648951, -0.00012969, -0.999979, -0.0684439,
            //     0.999908, 0.0119191, -0.0064906, -0.0958121,
            //     0., 0., 0., 1.;
            // Eigen::Matrix<double, 3, 3> R_test;
            // R_test << 0.0119197, -0.999929, 0.0000523,
            //     -0.00648951, -0.00012969, -0.999979,
            //     0.999908, 0.0119191, -0.0064906;
            // Eigen::Matrix<double, 3, 3> R_corr = correctRotationMatrix(R_test);
            // Eigen::Vector3d t_corr(0.0853154, -0.0684439, -0.0958121);
            // SE3 T_camera_lidar(R_corr, t_corr);
            // SE3 T_lidar_camera(T_camera_lidar.inverse());
            // T_camera_imu_matrix << 0.017014304328419078, -0.999823414494766, 0.0079783003357361, 0.7138061555049913,
            //     0.008227025113892006, -0.007839192351438318, -0.9999354294758499, -0.015324174578544,
            //     0.999821398803804, 0.01707884338309873, 0.008092193936149267, -0.14279853029864117,
            //     0.0, 0.0, 0.0, 1.0;
            // SE3 T_camera_imu(T_camera_imu_matrix);
            // // T_lidar_imu_matrix = T_lidar_camera_matrix * T_camera_imu_matrix;
            // SE3 T_lidar_imu = T_lidar_camera * T_camera_imu * T_small_adjust_se3;
            // LOG(INFO) << "T_lidar_imu: " << T_lidar_imu.matrix();
            // LOG(INFO)<<"roll pitch yaw: "<<T_lidar_imu.matrix().block<3,3>(0,0).eulerAngles(0,1,2).transpose()*180/M_PI<<std::endl;
            // SE3 mocap_pose(mocap.GetSE3());
            // static SE3 first_mocap_pose = mocap_pose;
            // SE3 T_mocap_imu(T_lidar_imu.inverse() * first_mocap_pose.inverse() * mocap_pose * T_lidar_imu);
            // mocap_convert.SetSE3(T_mocap_imu);




            /// 去掉原点
            if (!first_mocap_set) {
                origin = mocap_convert.position_;
                first_mocap_set = true;
            }
            mocap_convert.position_ -= origin;
            gins.AddMoCap(mocap_convert);

            auto state = gins.GetState();
            save_result(fout, state);
            if (ui) {
                ui->UpdateNavState(state);
                usleep(1e3);
            }
            mocap_inited = true;
        })
        // .SetOdomProcessFunc([&](const sad::Odom& odom) {
        //     imu_init.AddOdom(odom);

        //     if (imu_inited && mocap_inited) {
        //         gins.AddOdom(odom);
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