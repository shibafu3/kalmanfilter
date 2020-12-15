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

class ExtendedKalmanFilter {
public:
    typedef std::vector<std::function<double(Eigen::MatrixXd)>> FunctionVector;
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

    FunctionVector f;
    FunctionVector h;
    FunctionVector df;
    FunctionVector dh;

    Eigen::MatrixXd Calcx_kk1() {
        for (int i = 0; i < f.size(); ++i) {
            x_kk1(i) = f[i](x_k1);
        }
        return x_kk1;
    }
    Eigen::MatrixXd CalcA() {
        for (int row = 0; row < A.rows(); ++row) {
            for (int col = 0; col < A.cols(); ++col) {
                A(row, col) = df[row * A.cols() + col](x_kk1);
            }
        }
        At = A.transpose();
        return A;
    }
    Eigen::MatrixXd CalcC() {
        for (int row = 0; row < C.rows(); ++row) {
            for (int col = 0; col < C.cols(); ++col) {
                C(row, col) = dh[row * C.cols() + col](x_kk1);
            }
        }
        Ct = C.transpose();
        return C;
    }
public:
    ExtendedKalmanFilter() {}
    FunctionVector SetStateSpaceModelFunction(FunctionVector non_liner_state_function) {
        f = non_liner_state_function;
        A.resize(f.size(), f.size());
        I = Eigen::MatrixXd::Identity(A.rows(), A.cols());
        return f;
    }
    FunctionVector SetObservationFunction(FunctionVector non_liner_obsevation_function) {
        h = non_liner_obsevation_function;
        C.resize(h.size(), f.size());
        return h;
    }
    FunctionVector SetStateSpaceModelCoefficientJacobian(FunctionVector state_space_model_coefficient_jacobian) {
        df = state_space_model_coefficient_jacobian;
        return df;
    }
    FunctionVector SetObservationFunctionJacobian(FunctionVector obsevation_jacobian) {
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
        x_k1 = initial_state_matrix;
        x_kk1 = x_k1;
        return x_k1;
    }
    Eigen::MatrixXd SetInitialKyobunsanMatrix(Eigen::MatrixXd initial_kyobunsan_matrix) {
        P_k1 = initial_kyobunsan_matrix;
        return P_k1;
    }
    Eigen::MatrixXd Update(Eigen::MatrixXd obsevation_data) {
        Calcx_kk1();
        CalcA();
        CalcC();
        P_kk1 = A * P_k1 * At + B * Q * Bt;
        G = P_kk1 * Ct * ((C * P_kk1 * Ct) + R).inverse();
        x_k = x_kk1 + G * (obsevation_data - (C * x_kk1));
        P_k = (I - G * C) * P_kk1;
        x_k1 = x_k;
        P_k1 = P_k;
        return x_k1;
    }
    void Print() {
        std::cout << "updateing" << std::endl;
        std::cout << "f.size() " << f.size() << std::endl << std::endl;
        std::cout << "h.size() " << h.size() << std::endl << std::endl;
        std::cout << "df.size() " << df.size() << std::endl << std::endl;
        std::cout << "dh.size() " << dh.size() << std::endl << std::endl;
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

    auto Setf(FunctionVector non_liner_state_function) {
        return SetStateSpaceModelFunction(non_liner_state_function);
    }
    auto Setdf(FunctionVector non_liner_obsevation_function) {
        return SetObservationFunction(non_liner_obsevation_function);
    }
    auto Seth(FunctionVector state_space_model_coefficient_jacobian) {
        return SetStateSpaceModelCoefficientJacobian(state_space_model_coefficient_jacobian);
    }
    auto Setdh(FunctionVector obsevation_jacobian) {
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
public:
    typedef std::vector<std::function<double(Eigen::MatrixXd)>> FunctionVector;
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

    FunctionVector f;
    FunctionVector h;
    FunctionVector df;
    FunctionVector dh;

    Eigen::MatrixXd Calcx_kk1() {
        for (int i = 0; i < f.size(); ++i) {
            x_kk1(i) = f[i](x_k1);
        }
        return x_kk1;
    }
    Eigen::MatrixXd CalcA() {
        for (int row = 0; row < A.rows(); ++row) {
            for (int col = 0; col < A.cols(); ++col) {
                A(row, col) = df[row * A.cols() + col](x_kk1);
            }
        }
        At = A.transpose();
        return A;
    }
    Eigen::MatrixXd CalcC() {
        for (int row = 0; row < C.rows(); ++row) {
            for (int col = 0; col < C.cols(); ++col) {
                C(row, col) = dh[row * C.cols() + col](x_kk1);
            }
        }
        Ct = C.transpose();
        return C;
    }
    Eigen::MatrixXd F(Eigen::MatrixXd x) {
        Eigen::MatrixXd y = Eigen::MatrixXd::Zero(x.rows(), x.cols());
        for (size_t i = 0; i < f.size(); ++i) {
            y(i) = f[i](x);
        }
        return y;
    }
    Eigen::MatrixXd dF(Eigen::MatrixXd x) {
        Eigen::MatrixXd y = Eigen::MatrixXd::Zero(A.rows(), A.cols());
        for (size_t row = 0; row < y.rows(); ++row) {
            for (size_t col = 0; col < y.cols(); ++col) {
                y(row, col) = df[row * y.cols() + col](x);
            }
        }
        return y;
    }
    Eigen::MatrixXd H(Eigen::MatrixXd x) {
        Eigen::MatrixXd y = Eigen::MatrixXd::Zero(h.size(), 1);
        for (size_t i = 0; i < h.size(); ++i) {
            y(i) = h[i](x);
        }
        return y;
    }
    Eigen::MatrixXd dH(Eigen::MatrixXd x) {
        Eigen::MatrixXd y = Eigen::MatrixXd::Zero(C.rows(), C.cols());
        for (size_t row = 0; row < y.rows(); ++row) {
            for (size_t col = 0; col < y.cols(); ++col) {
                y(row, col) = dh[row * y.cols() + col](x);
            }
        }
        return y;
    }
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
        for (size_t i = 1; i < n+1; ++i) {
            w[i] = 1.0 / (2.0*(n + k));
        }
        return w;
    }
    std::vector<Eigen::MatrixXd> TransformSigmaPointsF(std::vector<Eigen::MatrixXd> X) {
        std::vector<Eigen::MatrixXd> Y(X.size());
        for (size_t i = 0; i < X.size(); ++i) {
            Y[i] = F(X[i]);
        }
        return Y;
    }
    std::vector<Eigen::MatrixXd> TransformSigmaPointsH(std::vector<Eigen::MatrixXd> X) {
        std::vector<Eigen::MatrixXd> Y(X.size());
        for (size_t i = 0; i < X.size(); ++i) {
            Y[i] = H(X[i]);
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
        Eigen::MatrixXd P = Eigen::MatrixXd::Zero(transformed_sigmapoints_1[0].rows(), transformed_sigmapoints_1[0].cols());
        for (size_t i = 0; i < 2*n+1; ++i) {
            P += weights[i] * (transformed_sigmapoints_1[i] - transformed_average_1) * (transformed_sigmapoints_2[i] - transformed_average_2).transpose();
        }
        return P;
    }
public:
    UnscentedKalmanFilter() {}
    double Setk(double scaling_parameter) {
        k = scaling_parameter;
        return k;
    }
    FunctionVector SetStateSpaceModelFunction(FunctionVector non_liner_state_function) {
        f = non_liner_state_function;
        A.resize(f.size(), f.size());
        I = Eigen::MatrixXd::Identity(A.rows(), A.cols());
        n = f.size();
        return f;
    }
    FunctionVector SetObservationFunction(FunctionVector non_liner_obsevation_function) {
        h = non_liner_obsevation_function;
        C.resize(h.size(), f.size());
        return h;
    }
    FunctionVector SetStateSpaceModelCoefficientJacobian(FunctionVector state_space_model_coefficient_jacobian) {
        df = state_space_model_coefficient_jacobian;
        return df;
    }
    FunctionVector SetObservationFunctionJacobian(FunctionVector obsevation_jacobian) {
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
        x_k1 = initial_state_matrix;
        x_kk1 = x_k1;
        return x_k1;
    }
    Eigen::MatrixXd SetInitialKyobunsanMatrix(Eigen::MatrixXd initial_kyobunsan_matrix) {
        P_k1 = initial_kyobunsan_matrix;
        return P_k1;
    }
    Eigen::MatrixXd Update(Eigen::MatrixXd obsevation_data){
        auto X = GetSigmaPoints(x_k1, P_k1, k);
        auto w = GetWeights(k);

        auto X_ = TransformSigmaPointsF(X);
        x_kk1 = GetTransformedAverage(w, X_);
        P_kk1 = GetTransformedVarianceCovarianceMatrix(w, X_, x_kk1, X_, x_kk1) + B * Q * Bt;

        X_ = GetSigmaPoints(x_kk1, P_kk1, k);
        auto Y_ = TransformSigmaPointsH(X_);
        auto y_ = GetTransformedAverage(w, Y_);
        auto Pyy_ = GetTransformedVarianceCovarianceMatrix(w, Y_, y_, Y_, y_);
        auto Pxy_ = GetTransformedVarianceCovarianceMatrix(w, X_, x_kk1, Y_, y_);

        auto g = Pxy_ * (Pyy_ + R).inverse();
        x_k = x_kk1 + g * (obsevation_data - y_);
        P_k = P_kk1 - g * Pxy_.transpose();
        x_k1 = x_k;
        P_k1 = P_k;

        return x_k;
    }
    void Print() {
        std::cout << "updateing" << std::endl;
        std::cout << "f.size() " << f.size() << std::endl << std::endl;
        std::cout << "h.size() " << h.size() << std::endl << std::endl;
        std::cout << "df.size() " << df.size() << std::endl << std::endl;
        std::cout << "dh.size() " << dh.size() << std::endl << std::endl;
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

    auto Setf(FunctionVector non_liner_state_function) {
        return SetStateSpaceModelFunction(non_liner_state_function);
    }
    auto Setdf(FunctionVector non_liner_obsevation_function) {
        return SetObservationFunction(non_liner_obsevation_function);
    }
    auto Seth(FunctionVector state_space_model_coefficient_jacobian) {
        return SetStateSpaceModelCoefficientJacobian(state_space_model_coefficient_jacobian);
    }
    auto Setdh(FunctionVector obsevation_jacobian) {
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


#endif