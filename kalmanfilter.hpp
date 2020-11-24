#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include <iostream>
#include <Eigen/Core>
#include <Eigen/LU>

class KalmanFilter {
    double dt;
    Eigen::MatrixXd A;
    Eigen::MatrixXd At;

    Eigen::MatrixXd B;
    Eigen::MatrixXd Bt;

    Eigen::MatrixXd C;
    Eigen::MatrixXd Ct;

    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;

    Eigen::MatrixXd I;

    Eigen::MatrixXd x_k1;
    Eigen::MatrixXd x_kk1;

    Eigen::MatrixXd P_k1;
    Eigen::MatrixXd P_kk1;

    Eigen::MatrixXd G;
    Eigen::MatrixXd x_k;
    Eigen::MatrixXd P_k;
public:
    KalmanFilter() {}
    int Init() {
        dt = 0.1;
        return 0;
    }
    Eigen::MatrixXd SetStateSpaceModelCoefficientMatrix(Eigen::MatrixXd state_space_model_coefficient_matrix) {
        A = state_space_model_coefficient_matrix;
        At = A.transpose();

        I = Eigen::MatrixXd::Identity(A.rows(), A.cols());
        return A;
    }
    Eigen::MatrixXd SetSystemMatrix(Eigen::MatrixXd system_matrix) {
        B = system_matrix;
        Bt = B.transpose();
        return B;
    }
    Eigen::MatrixXd SetSystemNoiseMatrix(Eigen::MatrixXd system_noise_matrix) {
        Q = system_noise_matrix;
        return Q;
    }
    Eigen::MatrixXd SetObservationMatrix(Eigen::MatrixXd observation_matrix) {
        C = observation_matrix;
        Ct = C.transpose();
        return C;
    }
    Eigen::MatrixXd SetObservationNoiseMatrix(Eigen::MatrixXd observation_noise_matrix) {
        R = observation_noise_matrix;
        return R;
    }
    Eigen::MatrixXd SetInitialStateMatrix(Eigen::MatrixXd initial_state_matrix) {
        x_k1 = initial_state_matrix;
        return x_k1;
    }
    Eigen::MatrixXd SetInitialKyobunsanMatrix(Eigen::MatrixXd initial_kyobunsan_matrix) {
        P_k1 = initial_kyobunsan_matrix;
        return P_k1;
    }
    Eigen::Vector3d Update(Eigen::MatrixXd sensor_data, double delta_time) {
        x_kk1 = A * x_k1;
        P_kk1 = A * P_k1 * At + B * Q * Bt;
        G = P_kk1 * Ct * ((C * P_kk1 * Ct) + R).inverse();
        x_k = x_kk1 + G * (sensor_data - (C * x_kk1));
        P_k = (I - G * C) * P_kk1;
        x_k1 = x_k;
        P_k1 = P_k;
        return x_k1;
    }
};

#endif