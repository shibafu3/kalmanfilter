#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include <Eigen/Core>

class KalmanFilter {
    double dt;
    Eigen::Matrix3d A;
    Eigen::Matrix3d At;

    Eigen::Matrix3d B;
    Eigen::Matrix3d Bt;

    Eigen::MatrixXd C;
    Eigen::Vector3d Ct;

    Eigen::Matrix3d Q;
    double R = 1;

    Eigen::Matrix3d I;

    Eigen::Vector3d x_k1;
    Eigen::Vector3d x_kk1;

    Eigen::Matrix3d P_k1;
    Eigen::Matrix3d P_kk1;

    Eigen::Vector3d G;
    Eigen::Vector3d x_k;
    Eigen::Matrix3d P_k;
public:
    KalmanFilter() {}
    int Init() {
        I << 1, 0, 0,
             0, 1, 0,
             0, 0, 1;

        dt = 0.1;

        A << 1, dt, dt*dt/2.0,
             0,  1,        dt,
             0,  0,         1;
        At = A.transpose();

        B << 1, 0, 0,
             0, 1, 0,
             0, 0, 1;
        Bt = B.transpose();

        Ct << 1, 0, 0;
        C = Ct.transpose();

        Q << 1, 0, 0,
             0, 1, 0,
             0, 0, 1;

        R = 20;

        x_k1 << 10, 1, 0;
        P_k1 << 1, 0, 0,
                0, 1, 0,
                0, 0, 1;
        return 0;
    }
    Eigen::Vector3d Update(double sensor_data, double delta_time) {
        A << 1, delta_time, delta_time*delta_time/2.0,
             0,          1,                delta_time,
             0,          0,                         1;
        At = A.transpose();
        x_kk1 = A * x_k1;
        P_kk1 = A * P_k1 * At + B * Q * Bt;
        G = P_kk1 * Ct / ((C * P_kk1 * Ct)(0, 0) + R);
        x_k = x_kk1 + G * (sensor_data - (C * x_kk1)(0, 0));
        P_k = (I - G * C) * P_kk1;
        x_k1 = x_k;
        P_k1 = P_k;
        return x_k1;
    }
};

#endif