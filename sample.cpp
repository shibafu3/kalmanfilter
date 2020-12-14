#include <iostream>
#include <cmath>
#include <random>
#include "kalmanfilter.hpp"

using namespace std;
using namespace Eigen;



int main() {
    {
        KalmanFilter kf;

        double dt = 0.1;
        Eigen::Matrix<double, 3, 3> A;
        A << 1, dt, dt*dt/2.0,
            0,  1,        dt,
            0,  0,         1;
        Eigen::Matrix<double, 3, 3> B;
        B << 0.5, 0, 0,
             0, 0.5, 0,
             0, 0, 0.5;
        Eigen::Matrix<double, 3, 3> Q;
        Q << 1, 0, 0,
            0, 1, 0,
            0, 0, 1;
        Eigen::Matrix<double, 1, 3> C;
        C << 1, 0, 0;
        Eigen::Matrix<double, 1, 1> R;
        R << 20;
        Eigen::Matrix<double, 3, 1> initX;
        initX << 5,
                 1,
                 1;
        Eigen::Matrix<double, 3, 3> initKyobunsan;
        initKyobunsan << 1, 0, 0,
                         0, 1, 0,
                         0, 0, 1;
        kf.SetStateSpaceModelCoefficientMatrix(A);
        kf.SetSystemMatrix(B);
        kf.SetSystemNoiseMatrix(Q);
        kf.SetObservationMatrix(C);
        kf.SetObservationNoiseMatrix(R);
        kf.SetInitialStateMatrix(initX);
        kf.SetInitialKyobunsanMatrix(initKyobunsan);

        Eigen::Matrix<double, 1, 1> data;

        random_device seed_gen;
        default_random_engine engine(seed_gen());
        normal_distribution<> dist(0.0, 1.0);

        FILE *gplot;
        gplot = popen("gnuplot -persist","w");
        fprintf(gplot, "plot '-', '-', '-'\n");

        int sample_nums = 100;
        double noizu[sample_nums] = {};
        for (int i = 0; i < sample_nums; ++i) {
            fprintf(gplot,"%d\t%f\n", i, i*0.1*i*0.1*0.5 * 1 + 1*i*0.1 + 1.0);
        }
        fprintf(gplot,"e\n");
        for (int i = 0; i < sample_nums; ++i) {
            noizu[i] = i*0.1*i*0.1*0.5 * 1 + 1*i*0.1 + 1.0 + dist(engine) * 1.0;
            fprintf(gplot,"%d\t%f\n", i, noizu[i]);
        }
        fprintf(gplot,"e\n");
        for (int i = 0; i < sample_nums; ++i) {
            data << noizu[i];
            fprintf(gplot,"%d\t%f\n", i, kf.Update(data, 0.1)(0, 0));
        }
        fprintf(gplot,"e\n");
        fflush(gplot);
        pclose(gplot);
    }


    {
        ExtendedKalmanFilter ekf;

        Eigen::Matrix<double, 1, 1> x;
        x << 10;

        std::vector<std::function<double(Eigen::MatrixXd)>> f(1);
        f[0] = [](Eigen::MatrixXd x) -> double { return x(0) + 3*std::cos(x(0) / 10.0); };
        ExtendedKalmanFilter::FunctionVector df(f.size() * x.size());
        df[0] = [](Eigen::MatrixXd x) -> double { return 1 - 3.0/10.0 * std::sin(x(0) / 10.0); };
        ExtendedKalmanFilter::FunctionVector h(1);
        h[0] = [](Eigen::MatrixXd x) -> double { return x(0); };
        ExtendedKalmanFilter::FunctionVector dh(h.size() * x.size());
        dh[0] = [](Eigen::MatrixXd x) -> double { return 1; };

        Eigen::Matrix<double, 1, 1> B;
        B << 1.0;
        Eigen::Matrix<double, 1, 1> Q;
        Q << 1.0;
        Eigen::Matrix<double, 1, 1> R;
        R << 100.0;
        Eigen::Matrix<double, 1, 1> P;
        P << 1;

        ekf.SetStateSpaceModelFunction(f);
        ekf.SetStateSpaceModelCoefficientJacobian(df);
        ekf.SetObservationFunction(h);
        ekf.SetObservationFunctionJacobian(dh);
        ekf.SetSystemMatrix(B);
        ekf.SetSystemNoiseMatrix(Q);
        ekf.SetObservationNoiseMatrix(R);
        ekf.SetInitialStateMatrix(x);
        ekf.SetInitialKyobunsanMatrix(P);

        Eigen::Matrix<double, 1, 1> data;

        random_device seed_gen;
        default_random_engine engine(seed_gen());
        normal_distribution<> dist(0.0, 1.0);

        int sample_nums = 50;
        double Y[sample_nums] = {};
        double shin[sample_nums] = {};
        double esti[sample_nums] = {};
        double xk1 = 10;
        for (int i = 0; i < sample_nums; ++i) {
            Y[i] = xk1 + dist(engine) * 1.0;
            shin[i] = xk1;
            data << Y[i];
            esti[i] = ekf.Update(data)(0, 0);
            cout << shin[i] << " " << Y[i] << " " << esti[i] << endl;
            xk1 = xk1 + 3*cos(xk1 / 10.0);
        }

        FILE *gplot;
        gplot = popen("gnuplot -persist","w");
        fprintf(gplot, "plot '-', '-', '-'\n");

        for (int i = 0; i < sample_nums; ++i) {
            fprintf(gplot,"%d\t%f\n", i, shin[i]);
        }
        fprintf(gplot,"e\n");
        for (int i = 0; i < sample_nums; ++i) {
            fprintf(gplot,"%d\t%f\n", i, Y[i]);
        }
        fprintf(gplot,"e\n");
        for (int i = 0; i < sample_nums; ++i) {
            fprintf(gplot,"%d\t%f\n", i, esti[i]);
        }
        fprintf(gplot,"e\n");
        fflush(gplot);
        pclose(gplot);
    }

    {
        UnscentedKalmanFilter ukf;

        Eigen::Matrix<double, 1, 1> x;
        x << 10;

        std::vector<std::function<double(Eigen::MatrixXd)>> f(1);
        f[0] = [](Eigen::MatrixXd x) -> double { return x(0) + 3*std::cos(x(0) / 10.0); };
        ExtendedKalmanFilter::FunctionVector df(f.size() * x.size());
        df[0] = [](Eigen::MatrixXd x) -> double { return 1 - 3.0/10.0 * std::sin(x(0) / 10.0); };
        ExtendedKalmanFilter::FunctionVector h(1);
        h[0] = [](Eigen::MatrixXd x) -> double { return x(0); };
        ExtendedKalmanFilter::FunctionVector dh(h.size() * x.size());
        dh[0] = [](Eigen::MatrixXd x) -> double { return 1; };

        Eigen::Matrix<double, 1, 1> B;
        B << 1.0;
        Eigen::Matrix<double, 1, 1> Q;
        Q << 1.0;
        Eigen::Matrix<double, 1, 1> R;
        R << 100.0;
        Eigen::Matrix<double, 1, 1> P;
        P << 1;

        ukf.Setk(3);
        ukf.SetStateSpaceModelFunction(f);
        ukf.SetStateSpaceModelCoefficientJacobian(df);
        ukf.SetObservationFunction(h);
        ukf.SetObservationFunctionJacobian(dh);
        ukf.SetSystemMatrix(B);
        ukf.SetSystemNoiseMatrix(Q);
        ukf.SetObservationNoiseMatrix(R);
        ukf.SetInitialStateMatrix(x);
        ukf.SetInitialKyobunsanMatrix(P);

        Eigen::Matrix<double, 1, 1> data;

        random_device seed_gen;
        default_random_engine engine(seed_gen());
        normal_distribution<> dist(0.0, 1.0);

        int sample_nums = 50;
        double Y[sample_nums] = {};
        double shin[sample_nums] = {};
        double esti[sample_nums] = {};
        double xk1 = 10;
        for (int i = 0; i < sample_nums; ++i) {
            Y[i] = xk1 + dist(engine) * 1.0;
            shin[i] = xk1;
            data << Y[i];
            esti[i] = ukf.Update(data)(0, 0);
            cout << shin[i] << " " << Y[i] << " " << esti[i] << endl;
            xk1 = xk1 + 3*cos(xk1 / 10.0);
        }

        FILE *gplot;
        gplot = popen("gnuplot -persist","w");
        fprintf(gplot, "plot '-', '-', '-'\n");
        for (int i = 0; i < sample_nums; ++i) {
            fprintf(gplot,"%d\t%f\n", i, shin[i]);
        }
        fprintf(gplot,"e\n");
        for (int i = 0; i < sample_nums; ++i) {
            fprintf(gplot,"%d\t%f\n", i, Y[i]);
        }
        fprintf(gplot,"e\n");
        for (int i = 0; i < sample_nums; ++i) {
            fprintf(gplot,"%d\t%f\n", i, esti[i]);
        }
        fprintf(gplot,"e\n");
        fflush(gplot);
        pclose(gplot);
    }
}