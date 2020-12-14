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
        C.resize(h.size(), h.size());
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
        for (size_t i = 0; i < f.size(); ++i) {
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
public:
    UnscentedKalmanFilter() {}
    FunctionVector SetStateSpaceModelFunction(FunctionVector non_liner_state_function) {
        f = non_liner_state_function;
        A.resize(f.size(), f.size());
        I = Eigen::MatrixXd::Identity(A.rows(), A.cols());
        return f;
    }
    FunctionVector SetObservationFunction(FunctionVector non_liner_obsevation_function) {
        h = non_liner_obsevation_function;
        C.resize(h.size(), h.size());
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
    Eigen::MatrixXd GetSigmaPoint(unsigned char k){
        Eigen::MatrixXd ave = x_k1;
        Eigen::MatrixXd avey = x_k1;
        Eigen::MatrixXd Py = P_k1;
        Eigen::MatrixXd rootP = P_k1.llt().matrixL(); 
        unsigned char n = x_k1.size();
        std::vector<Eigen::MatrixXd> X(2*n + 1);
        std::vector<double> w(2*n + 1);
        std::vector<Eigen::MatrixXd> Y(2*n + 1);

        X[0] = ave;
        for (unsigned char i = 1; i < n+1; ++i) {
            X[i] = ave + std::sqrt(n + k) * rootP.col(i-1);
            X[n + i] =  ave + std::sqrt(n + k) * rootP.col(i-1);
        }
        w[0] = k / (n + k);
        for (unsigned char i = 1; i < n+1; ++i) {
            w[i] = 1.0 / (2.0*(n + k));
        }

        for (int i = 0; i < X.size(); ++i) {
            Y[i] = X[i];
        }
        for (int j = 0; j < 2*n+1; ++j) {
            for (int i = 0; i < f.size(); ++i) {
                Y[j](i) = f[i](X[j]);
            }
        }

        for (unsigned int i = 0; i < 2*n+1; ++i) {
            avey += w[i] * Y[i];
        }
        for (unsigned int i = 0; i < 2*n+1; ++i) {
            Py += w[i] * (Y[i] - avey) * (Y[i] - avey).transpose();
        }
        return Py;
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


#endif