#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include <iostream>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <cmath>
#include <vector>
#include <functional>

class KalmanFilter {
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
    KalmanFilter(const KalmanFilter &obj) {
        A     = obj.A;
        At    = obj.At;
        B     = obj.B;
        Bt    = obj.Bt;
        C     = obj.C;
        Ct    = obj.Ct;
        Q     = obj.Q;
        R     = obj.R;
        I     = obj.I;
        x_k1  = obj.x_k1;
        x_kk1 = obj.x_kk1;
        P_k1  = obj.P_k1;
        P_kk1 = obj.P_kk1;
        G     = obj.G;
        x_k   = obj.x_k;
        P_k   = obj.P_k;
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
    Eigen::Vector3d PredictStep() {
        x_kk1 = A * x_k1;
        P_kk1 = A * P_k1 * At + B * Q * Bt;

        P_k1 = P_kk1;
        x_k1 = x_kk1;

        return x_k1;
    }
    Eigen::Vector3d FilteringStep(Eigen::MatrixXd obsevation_data) {
        G = P_kk1 * Ct * ((C * P_kk1 * Ct) + R).inverse();
        x_k = x_kk1 + G * (obsevation_data - (C * x_kk1));
        P_k = (I - G * C) * P_kk1;

        x_k1 = x_k;
        P_k1 = P_k;

        return x_k1;
    }
    Eigen::Vector3d Update(Eigen::MatrixXd obsevation_data) {
        PredictStep();
        FilteringStep(obsevation_data);
        return x_k1;
    }
};

class ExtendedKalmanFilter {
private:
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

    std::function<Eigen::MatrixXd(Eigen::MatrixXd)> f;
    std::function<Eigen::MatrixXd(Eigen::MatrixXd)> h;
    std::function<Eigen::MatrixXd(Eigen::MatrixXd)> df;
    std::function<Eigen::MatrixXd(Eigen::MatrixXd)> dh;

public:
    ExtendedKalmanFilter() {}
    ExtendedKalmanFilter(const ExtendedKalmanFilter &obj) {
        A     = obj.A;
        At    = obj.At;
        B     = obj.B;
        Bt    = obj.Bt;
        C     = obj.C;
        Ct    = obj.Ct;
        Q     = obj.Q;
        R     = obj.R;
        I     = obj.I;
        x_k1  = obj.x_k1;
        x_kk1 = obj.x_kk1;
        P_k1  = obj.P_k1;
        P_kk1 = obj.P_kk1;
        G     = obj.G;
        x_k   = obj.x_k;
        P_k   = obj.P_k;
        f     = obj.f;
        h     = obj.h;
        df    = obj.df;
        dh    = obj.dh;
    }
    std::function<Eigen::MatrixXd(Eigen::MatrixXd)> SetStateSpaceModelFunction(std::function<Eigen::MatrixXd(Eigen::MatrixXd)> non_liner_state_function) {
        f = non_liner_state_function;
        return f;
    }
    std::function<Eigen::MatrixXd(Eigen::MatrixXd)> SetObservationFunction(std::function<Eigen::MatrixXd(Eigen::MatrixXd)> non_liner_obsevation_function) {
        h = non_liner_obsevation_function;
        return h;
    }
    std::function<Eigen::MatrixXd(Eigen::MatrixXd)> SetStateSpaceModelCoefficientJacobian(std::function<Eigen::MatrixXd(Eigen::MatrixXd)> state_space_model_coefficient_jacobian) {
        df = state_space_model_coefficient_jacobian;
        return df;
    }
    std::function<Eigen::MatrixXd(Eigen::MatrixXd)> SetObservationFunctionJacobian(std::function<Eigen::MatrixXd(Eigen::MatrixXd)> obsevation_jacobian) {
        dh = obsevation_jacobian;
        return dh;
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
    Eigen::MatrixXd SetObservationJacobian(Eigen::MatrixXd observation_jacobian) {
        C = observation_jacobian;
        Ct = C.transpose();
        return C;
    }
    Eigen::MatrixXd SetObservationNoiseMatrix(Eigen::MatrixXd observation_noise_matrix) {
        R = observation_noise_matrix;
        return R;
    }
    Eigen::MatrixXd SetInitialStateMatrix(Eigen::MatrixXd initial_state_matrix) {
        I = Eigen::MatrixXd::Identity(initial_state_matrix.rows(), initial_state_matrix.rows());
        x_k1 = initial_state_matrix;
        x_kk1 = x_k1;
        return x_k1;
    }
    Eigen::MatrixXd SetInitialKyobunsanMatrix(Eigen::MatrixXd initial_kyobunsan_matrix) {
        P_k1 = initial_kyobunsan_matrix;
        return P_k1;
    }
    Eigen::MatrixXd PredictStep() {
        x_kk1 = f(x_k1);
        A = df(x_k1); At = A.transpose();
        C = dh(x_kk1); Ct = C.transpose();
        P_kk1 = A * P_k1 * At + B * Q * Bt;

        x_k1 = x_kk1;
        P_k1 = P_kk1;

        return x_k1;
    }
    Eigen::MatrixXd FilteringStep(Eigen::MatrixXd obsevation_data) {
        G = P_kk1 * Ct * ((C * P_kk1 * Ct) + R).inverse();
        x_k = x_kk1 + G * (obsevation_data - h(x_kk1));
        P_k = (I - G * C) * P_kk1;

        x_k1 = x_k;
        P_k1 = P_k;

        return x_k1;
    }
    Eigen::MatrixXd Update(Eigen::MatrixXd obsevation_data) {
        PredictStep();
        FilteringStep(obsevation_data);
        return x_k1;
    }
    void Print() {
        std::cout << "updateing" << std::endl;
        std::cout << "B " << std::endl << B << std::endl << std::endl;
        std::cout << "Bt " << std::endl << Bt << std::endl << std::endl;
        std::cout << "Q " << std::endl << Q << std::endl << std::endl;
        std::cout << "R " << std::endl << R << std::endl << std::endl;
        std::cout << "x_k1 " << std::endl << x_k1 << std::endl << std::endl;
        std::cout << "x_kk1 " << std::endl << x_kk1 << std::endl << std::endl;
        std::cout << "A " << std::endl << A << std::endl << std::endl;
        std::cout << "At " << std::endl << At << std::endl << std::endl;
        std::cout << "C " << std::endl << C << std::endl << std::endl;
        std::cout << "Ct " << std::endl << Ct << std::endl << std::endl;
        std::cout << "P_k1 " << std::endl << P_k1 << std::endl << std::endl;
        std::cout << "P_kk1 " << std::endl << P_kk1 << std::endl << std::endl;
        std::cout << "G " << std::endl << G << std::endl << std::endl;
        std::cout << "x_k " << std::endl << x_k << std::endl << std::endl;
        std::cout << "P_k " << std::endl << P_k << std::endl << std::endl;
    }

    auto Setf(std::function<Eigen::MatrixXd(Eigen::MatrixXd)> non_liner_state_function) {
        return SetStateSpaceModelFunction(non_liner_state_function);
    }
    auto Seth(std::function<Eigen::MatrixXd(Eigen::MatrixXd)> non_liner_obsevation_function) {
        return SetObservationFunction(non_liner_obsevation_function);
    }
    auto Setdf(std::function<Eigen::MatrixXd(Eigen::MatrixXd)> state_space_model_coefficient_jacobian) {
        return SetStateSpaceModelCoefficientJacobian(state_space_model_coefficient_jacobian);
    }
    auto Setdh(std::function<Eigen::MatrixXd(Eigen::MatrixXd)> obsevation_jacobian) {
        return SetObservationFunctionJacobian(obsevation_jacobian);
    }
    auto SetB(Eigen::MatrixXd system_matrix) {
        return SetSystemMatrix(system_matrix);
    }
    auto SetQ(Eigen::MatrixXd system_noise_matrix) {
        return SetSystemNoiseMatrix(system_noise_matrix);
    }
    auto SetC(Eigen::MatrixXd observation_jacobian) {
        return SetObservationJacobian(observation_jacobian);
    }
    auto SetR(Eigen::MatrixXd observation_noise_matrix) {
        return SetObservationNoiseMatrix(observation_noise_matrix);
    }
    auto Setx(Eigen::MatrixXd initial_state_matrix) {
        return SetInitialStateMatrix(initial_state_matrix);
    }
    auto SetP(Eigen::MatrixXd initial_kyobunsan_matrix) {
        return SetInitialKyobunsanMatrix(initial_kyobunsan_matrix);
    }
};

class UnscentedKalmanFilter {
private:
    Eigen::MatrixXd A;
    Eigen::MatrixXd At;

    Eigen::MatrixXd B;
    Eigen::MatrixXd Bt;

    Eigen::MatrixXd C;
    Eigen::MatrixXd Ct;

    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;

    Eigen::MatrixXd I;

    unsigned char n;
    double k;
    Eigen::MatrixXd x_k1;
    Eigen::MatrixXd x_kk1;

    Eigen::MatrixXd P_k1;
    Eigen::MatrixXd P_kk1;

    Eigen::MatrixXd G;
    Eigen::MatrixXd x_k;
    Eigen::MatrixXd P_k;

    std::vector<double> w;
    std::vector<Eigen::MatrixXd> X;
    std::vector<Eigen::MatrixXd> X_;
    std::vector<Eigen::MatrixXd> Y_;
    Eigen::MatrixXd y_;
    Eigen::MatrixXd Pyy_;
    Eigen::MatrixXd Pxy_;

    std::function<Eigen::MatrixXd(Eigen::MatrixXd)> f;
    std::function<Eigen::MatrixXd(Eigen::MatrixXd)> h;

    std::vector<Eigen::MatrixXd> GetSigmaPoints(Eigen::MatrixXd x, Eigen::MatrixXd P, double k) {
        std::vector<Eigen::MatrixXd> X(2*n + 1);
        Eigen::MatrixXd sqrtP = P.llt().matrixL();

        X[0] = x;
        for (size_t i = 1; i < n+1; ++i) {
            X[i]     = X[0] + std::sqrt(n + k) * sqrtP.col(i-1);
            X[n + i] = X[0] - std::sqrt(n + k) * sqrtP.col(i-1);
        }
        return X;
    }
    std::vector<double> GetWeights(double k) {
        std::vector<double> w(2*n + 1);
        w[0] = k / (n + k);
        for (size_t i = 1; i < 2*n+1; ++i) {
            w[i] = 1.0 / (2.0*(n + k));
        }
        return w;
    }
    std::vector<Eigen::MatrixXd> TransformSigmaPointsF(std::vector<Eigen::MatrixXd> X) {
        std::vector<Eigen::MatrixXd> Y(X.size());
        for (size_t i = 0; i < X.size(); ++i) {
            Y[i] = f(X[i]);
        }
        return Y;
    }
    std::vector<Eigen::MatrixXd> TransformSigmaPointsH(std::vector<Eigen::MatrixXd> X) {
        std::vector<Eigen::MatrixXd> Y(X.size());
        for (size_t i = 0; i < X.size(); ++i) {
            Y[i] = h(X[i]);
        }
        return Y;
    }
    Eigen::MatrixXd GetTransformedAverage(std::vector<double>weights, std::vector<Eigen::MatrixXd> transformed_sigmapoints) {
        Eigen::MatrixXd average = Eigen::MatrixXd::Zero(transformed_sigmapoints[0].rows(), transformed_sigmapoints[0].cols());
        for (size_t i = 0; i < 2*n+1; ++i) {
            average += weights[i] * transformed_sigmapoints[i];
        }
        return average;
    }
    Eigen::MatrixXd GetTransformedVarianceCovarianceMatrix(std::vector<double>weights, std::vector<Eigen::MatrixXd> transformed_sigmapoints_1, Eigen::MatrixXd transformed_average_1, std::vector<Eigen::MatrixXd> transformed_sigmapoints_2, Eigen::MatrixXd transformed_average_2) {
        Eigen::MatrixXd P = weights[0] * (transformed_sigmapoints_1[0] - transformed_average_1) * (transformed_sigmapoints_2[0] - transformed_average_2).transpose();
        for (size_t i = 1; i < 2*n+1; ++i) {
            P += weights[i] * (transformed_sigmapoints_1[i] - transformed_average_1) * (transformed_sigmapoints_2[i] - transformed_average_2).transpose();
        }
        return P;
    }
public:
    UnscentedKalmanFilter() {}
    UnscentedKalmanFilter(const UnscentedKalmanFilter &obj) {
        A     = obj.A;
        At    = obj.At;
        B     = obj.B;
        Bt    = obj.Bt;
        C     = obj.C;
        Ct    = obj.Ct;
        Q     = obj.Q;
        R     = obj.R;
        I     = obj.I;
        n     = obj.n;
        k     = obj.k;
        x_k1  = obj.x_k1;
        x_kk1 = obj.x_kk1;
        P_k1  = obj.P_k1;
        P_kk1 = obj.P_kk1;
        G     = obj.G;
        x_k   = obj.x_k;
        P_k   = obj.P_k;
        f     = obj.f;
        h     = obj.h;
    }
    double Setk(double scaling_parameter) {
        k = scaling_parameter;
        return k;
    }
    std::function<Eigen::MatrixXd(Eigen::MatrixXd)> SetStateSpaceModelFunction(std::function<Eigen::MatrixXd(Eigen::MatrixXd)> non_liner_state_function) {
        f = non_liner_state_function;
        return f;
    }
    std::function<Eigen::MatrixXd(Eigen::MatrixXd)> SetObservationFunction(std::function<Eigen::MatrixXd(Eigen::MatrixXd)> non_liner_obsevation_function) {
        h = non_liner_obsevation_function;
        return h;
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
    Eigen::MatrixXd SetObservationNoiseMatrix(Eigen::MatrixXd observation_noise_matrix) {
        R = observation_noise_matrix;
        return R;
    }
    Eigen::MatrixXd SetInitialStateMatrix(Eigen::MatrixXd initial_state_matrix) {
        I = Eigen::MatrixXd::Identity(initial_state_matrix.rows(), initial_state_matrix.rows());
        n = initial_state_matrix.rows();
        x_k1 = initial_state_matrix;
        x_kk1 = x_k1;
        return x_k1;
    }
    Eigen::MatrixXd SetInitialKyobunsanMatrix(Eigen::MatrixXd initial_kyobunsan_matrix) {
        P_k1 = initial_kyobunsan_matrix;
        return P_k1;
    }
    Eigen::MatrixXd PredictStep(){
        X = GetSigmaPoints(x_k1, P_k1, k);
        w = GetWeights(k);
        X_ = TransformSigmaPointsF(X);
        x_kk1 = GetTransformedAverage(w, X_);
        P_kk1 = GetTransformedVarianceCovarianceMatrix(w, X_, x_kk1, X_, x_kk1) + B * Q * Bt;

        x_k1 = x_kk1;
        P_k1 = P_kk1;

        return x_k1;
    }
    Eigen::MatrixXd FilteringStep(Eigen::MatrixXd obsevation_data){
        X_ = GetSigmaPoints(x_kk1, P_kk1, k);
        Y_ = TransformSigmaPointsH(X_);
        y_ = GetTransformedAverage(w, Y_);
        Pyy_ = GetTransformedVarianceCovarianceMatrix(w, Y_, y_, Y_, y_);
        Pxy_ = GetTransformedVarianceCovarianceMatrix(w, X_, x_kk1, Y_, y_);

        G = Pxy_ * (Pyy_ + R).inverse();
        x_k = x_kk1 + G * (obsevation_data - y_);
        P_k = P_kk1 - G * Pxy_.transpose();

        x_k1 = x_k;
        P_k1 = P_k;

        return x_k1;
    }
    Eigen::MatrixXd Update(Eigen::MatrixXd obsevation_data){
        PredictStep();
        FilteringStep(obsevation_data);

        return x_k1;
    }
    void Print() {
        std::cout << "updateing" << std::endl;
        std::cout << "B " << std::endl << B << std::endl << std::endl;
        std::cout << "Bt " << std::endl << Bt << std::endl << std::endl;
        std::cout << "Q " << std::endl << Q << std::endl << std::endl;
        std::cout << "R " << std::endl << R << std::endl << std::endl;
        std::cout << "x_k1 " << std::endl << x_k1 << std::endl << std::endl;
        std::cout << "x_kk1 " << std::endl << x_kk1 << std::endl << std::endl;
        std::cout << "A " << std::endl << A << std::endl << std::endl;
        std::cout << "At " << std::endl << At << std::endl << std::endl;
        std::cout << "C " << std::endl << C << std::endl << std::endl;
        std::cout << "Ct " << std::endl << Ct << std::endl << std::endl;
        std::cout << "P_k1 " << std::endl << P_k1 << std::endl << std::endl;
        std::cout << "P_kk1 " << std::endl << P_kk1 << std::endl << std::endl;
        std::cout << "G " << std::endl << G << std::endl << std::endl;
        std::cout << "x_k " << std::endl << x_k << std::endl << std::endl;
        std::cout << "P_k " << std::endl << P_k << std::endl << std::endl;
    }

    auto Setf(std::function<Eigen::MatrixXd(Eigen::MatrixXd)> non_liner_state_function) {
        return SetStateSpaceModelFunction(non_liner_state_function);
    }
    auto Seth(std::function<Eigen::MatrixXd(Eigen::MatrixXd)> non_liner_obsevation_function) {
        return SetObservationFunction(non_liner_obsevation_function);
    }
    auto SetB(Eigen::MatrixXd system_matrix) {
        return SetSystemMatrix(system_matrix);
    }
    auto SetQ(Eigen::MatrixXd system_noise_matrix) {
        return SetSystemNoiseMatrix(system_noise_matrix);
    }
    auto SetR(Eigen::MatrixXd observation_noise_matrix) {
        return SetObservationNoiseMatrix(observation_noise_matrix);
    }
    auto Setx(Eigen::MatrixXd initial_state_matrix) {
        return SetInitialStateMatrix(initial_state_matrix);
    }
    auto SetP(Eigen::MatrixXd initial_kyobunsan_matrix) {
        return SetInitialKyobunsanMatrix(initial_kyobunsan_matrix);
    }
};


#endif