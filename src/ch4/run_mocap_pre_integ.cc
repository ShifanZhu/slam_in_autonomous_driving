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
DEFINE_string(txt_path, "./data/ch3/fasterlio_imu.txt", "数据文件路径");
// DEFINE_string(txt_path, "./data/ch3/eats_in_imu_imu.txt", "数据文件路径");
DEFINE_bool(with_ui, true, "是否显示图形界面");
DEFINE_bool(debug, false, "是否打印调试信息");

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

    std::ofstream fout("./data/ch4/mocap_preintg.txt");
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

            // Eigen::Matrix<double, 4, 4> T_marker_imu_matrix;

            // T_marker_imu_matrix << 0.00676736863383343, 0.999787658356272, 0.0194638362163181, -0.00683758,
            //     -0.0109922982027149, 0.0195374824414247, -0.999748696503313, -0.0195426,
            //     -0.999916682580102, 0.00655171567857404, 0.0111221814259042, -0.00116914,
            //     0, 0, 0, 1;
            // SE3 mocap_pose(mocap.GetSE3());
            // static SE3 first_mocap_pose = mocap_pose;
            // SE3 T_marker_imu(T_marker_imu_matrix);
            // SE3 T_mocap_imu(T_marker_imu.inverse() * first_mocap_pose.inverse() * mocap_pose * T_marker_imu);
            // // mocap_convert.SetSE3(T_mocap_imu);

            // Eigen::Matrix<double, 4, 4> T_adjust = Eigen::Matrix<double, 4, 4>::Identity();
            // double angleZ = -8 * M_PI / 2;
            // Eigen::AngleAxisd yawAngle(angleZ, Eigen::Vector3d::UnitY());
            // T_adjust.block<3, 3>(0, 0) = yawAngle.toRotationMatrix();
            // mocap_convert.SetSE3(SE3(T_adjust) * mocap.GetSE3());



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