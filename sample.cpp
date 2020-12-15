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
        fprintf(gplot, "plot '-' with lines, '-' with lines, '-' with lines\n");

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
        x << 5;

        std::vector<std::function<double(Eigen::MatrixXd)>> f(1);
        f[0] = [](Eigen::MatrixXd x) -> double { return  x(0) + 5*std::sin(0.1*x(0))*std::sin(0.1*x(0)); };
        ExtendedKalmanFilter::FunctionVector df(f.size() * x.size());
        df[0] = [](Eigen::MatrixXd x) -> double { return 1 + 5.0*2.0*0.1*cos(0.1*x(0)); };
        ExtendedKalmanFilter::FunctionVector h(1);
        h[0] = [](Eigen::MatrixXd x) -> double { return x(0); };
        ExtendedKalmanFilter::FunctionVector dh(h.size() * x.size());
        dh[0] = [](Eigen::MatrixXd x) -> double { return 1; };

        Eigen::Matrix<double, 1, 1> B;
        B << 1.0;
        Eigen::Matrix<double, 1, 1> Q;
        Q << 1.0;
        Eigen::Matrix<double, 1, 1> R;
        R << 10.0;
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

        int sample_nums = 100;
        double Y[sample_nums] = {};
        double shin[sample_nums] = {};
        double esti[sample_nums] = {};
        double xk1 = 1;
        for (int i = 0; i < sample_nums; ++i) {
            Y[i] = xk1 + dist(engine) * 1.0;
            shin[i] = xk1;
            data << Y[i];
            esti[i] = ekf.Update(data)(0, 0);
            cout << shin[i] << " " << Y[i] << " " << esti[i] << endl;
            xk1 = xk1 + 5*sin(0.1*xk1)*sin(0.1*xk1);
        }

        FILE *gplot;
        gplot = popen("gnuplot -persist","w");
        fprintf(gplot, "plot '-' with lines, '-' with lines, '-' with lines\n");

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
        double rho = 1.23, g = 9.81, eta = 6e3;
        double M = 3e4, a = 3e4;
        double T = 0.5;
        double EndTime = 30;
        vector<double> time; for (double i = 0; i < EndTime; i += 0.5) { time.push_back(i); }
        double N = EndTime/T+1;
        double n = 3;

        std::vector<std::function<double(Eigen::MatrixXd)>> f(3);
        f[0] = [&](Eigen::MatrixXd x) -> double { return x(0)+T*x(1); };
        f[1] = [&](Eigen::MatrixXd x) -> double { return x(1)+T*(0.5*rho*exp(-x(0)/eta)*x(1)*x(1)*x(2)-g); };
        f[2] = [&](Eigen::MatrixXd x) -> double { return x(2); };
        ExtendedKalmanFilter::FunctionVector h(1);
        h[0] = [&](Eigen::MatrixXd x) -> double { return sqrt(M*M+(x(0)-a)*(x(0)-a)); };

        Eigen::Matrix<double, 3, 1> B;
        B << 0.0,
             0.0,
             0.0;
        Eigen::Matrix<double, 1, 1> Q;
        Q << 0.0;
        Eigen::Matrix<double, 1, 1> R;
        R << 4e3;
        Eigen::Matrix<double, 3, 3> P;
        P << 9e3, 0, 0,
             0, 4e5, 0,
             0, 0, 0.4;

        Eigen::Matrix<double, 1, 1> data;

        random_device seed_gen;
        default_random_engine engine(seed_gen());
        normal_distribution<> dist(0.0, 1.0);


        vector<double> w;
        {
            w.push_back(34.0050556798653);
            w.push_back(115.985072259432);
            w.push_back(-142.862019325866);
            w.push_back(54.5286286038661);
            w.push_back(20.1604839368854);
            w.push_back(-82.7054697173963);
            w.push_back(-27.4227673152899);
            w.push_back(21.6694739272456);
            w.push_back(226.317694034193);
            w.push_back(175.154577016960);
            w.push_back(-85.3743462922003);
            w.push_back(191.945413558040);
            w.push_back(45.8785915027765);
            w.push_back(-3.98794033704800);
            w.push_back(45.2043103506628);
            w.push_back(-12.9631917450829);
            w.push_back(-7.85157798001215);
            w.push_back(94.2167493101250);
            w.push_back(89.1151657900561);
            w.push_back(89.6311181829716);
            w.push_back(42.4692076895187);
            w.push_back(-76.3681784110472);
            w.push_back(45.3621552821297);
            w.push_back(103.105132714875);
            w.push_back(30.9203569610492);
            w.push_back(65.4397318079126);
            w.push_back(45.9722523761265);
            w.push_back(-19.1912891526330);
            w.push_back(18.5860635072134);
            w.push_back(-49.7921364512133);
            w.push_back(56.1870731939673);
            w.push_back(-72.5470834783100);
            w.push_back(-67.6013034295741);
            w.push_back(-51.1971927463051);
            w.push_back(-186.212880613281);
            w.push_back(90.9711573359128);
            w.push_back(20.5668555644088);
            w.push_back(-47.7458591747763);
            w.push_back(86.6652892220897);
            w.push_back(-108.245802727049);
            w.push_back(-6.46638006354236);
            w.push_back(-15.2704517157739);
            w.push_back(20.1884068047652);
            w.push_back(19.7869150187634);
            w.push_back(-54.6998088256681);
            w.push_back(-1.90061085241126);
            w.push_back(-10.4278647815043);
            w.push_back(39.6996946495396);
            w.push_back(69.1441920366519);
            w.push_back(70.1566033613464);
            w.push_back(-54.6222005023258);
            w.push_back(4.89261851385345);
            w.push_back(-76.7875040770937);
            w.push_back(-70.4239703876904);
            w.push_back(-0.433189544967621);
            w.push_back(96.9320517037168);
            w.push_back(-48.6779464971270);
            w.push_back(23.4880584610197);
            w.push_back(-14.2672103156963);
            w.push_back(70.6678071244997);
            w.push_back(-68.8784738146142);
        }

        vector<Eigen::Matrix<double, 3, 1>> x(N);
        x[0] << 90000,
                -6000,
                0.003;
        vector<Eigen::Matrix<double, 1, 1>> y(N);
        y[0] << h[0](x[0]) + w[0];

        for (size_t k = 1; k < N; ++k) {
            x[k] << f[0](x[k-1]),
                    f[1](x[k-1]),
                    f[2](x[k-1]);
            y[k] << h[0](x[k]) + w[k];
        }
        vector<Eigen::Matrix<double, 3, 1>> x_ = x;

        ukf.Setk(0);
        ukf.SetStateSpaceModelFunction(f);
        ukf.SetObservationFunction(h);
        ukf.SetSystemMatrix(B);
        ukf.SetSystemNoiseMatrix(Q);
        ukf.SetObservationNoiseMatrix(R);
        ukf.SetInitialStateMatrix(x[0]);
        ukf.SetInitialKyobunsanMatrix(P);

        for (int i = 1; i < N; ++i) {
            x_[i] = ukf.Update(y[i]);
        }

        FILE *gplot;
        gplot = popen("gnuplot -persist","w");
        fprintf(gplot, "plot '-' with lines, '-' with lines\n");
        for (int i = 0; i < N; ++i) {
            fprintf(gplot,"%f\t%f\n", i*T, x[i](0));
        }
        fprintf(gplot,"e\n");
        for (int i = 0; i < N; ++i) {
            fprintf(gplot,"%f\t%f\n", i*T, x_[i](0));
        }
        fprintf(gplot,"e\n");
        fflush(gplot);
        pclose(gplot);

        gplot = popen("gnuplot -persist","w");
        fprintf(gplot, "plot '-' with lines, '-' with lines\n");
        for (int i = 0; i < N; ++i) {
            fprintf(gplot,"%f\t%f\n", i*T, x[i](1));
        }
        fprintf(gplot,"e\n");
        for (int i = 0; i < N; ++i) {
            fprintf(gplot,"%f\t%f\n", i*T, x_[i](1));
        }
        fprintf(gplot,"e\n");
        fflush(gplot);
        pclose(gplot);

        gplot = popen("gnuplot -persist","w");
        fprintf(gplot, "plot '-' with lines, '-' with lines\n");
        for (int i = 0; i < N; ++i) {
            fprintf(gplot,"%f\t%f\n", i*T, x[i](2));
        }
        fprintf(gplot,"e\n");
        for (int i = 0; i < N; ++i) {
            fprintf(gplot,"%f\t%f\n", i*T, x_[i](2));
        }
        fprintf(gplot,"e\n");
        fflush(gplot);
        pclose(gplot);
    }
}