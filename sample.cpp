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
            fprintf(gplot,"%d\t%f\n", i, kf.Update(data)(0, 0));
        }
        fprintf(gplot,"e\n");
        fflush(gplot);
        pclose(gplot);
    }

    // MTALAB例題コードと同じ例題
    {
        ExtendedKalmanFilter ekf;

        std::function<Eigen::MatrixXd(Eigen::MatrixXd)> f = [](Eigen::MatrixXd x) -> Eigen::MatrixXd {
            Eigen::MatrixXd fx = Eigen::MatrixXd::Zero(x.rows(), x.cols());
            fx(0) = x(0) + 3.0*cos(x(0)/10.0);
            return fx;
        };
        std::function<Eigen::MatrixXd(Eigen::MatrixXd)> df = [](Eigen::MatrixXd x) -> Eigen::MatrixXd {
            Eigen::MatrixXd dfx = Eigen::MatrixXd::Zero(x.rows(), x.cols());
            dfx(0) = 1 - 3.0/10.0*sin(x(0)/10.0);
            return dfx;
        };
        std::function<Eigen::MatrixXd(Eigen::MatrixXd)> h = [](Eigen::MatrixXd x) -> Eigen::MatrixXd {
            Eigen::MatrixXd hx = Eigen::MatrixXd::Zero(x.rows(), x.cols());
            hx(0) = x(0)*x(0)*x(0);
            return hx;
        };
        std::function<Eigen::MatrixXd(Eigen::MatrixXd)> dh = [](Eigen::MatrixXd x) -> Eigen::MatrixXd {
            Eigen::MatrixXd dhx = Eigen::MatrixXd::Zero(x.rows(), x.cols());
            dhx(0) = 3.0*x(0)*x(0);
            return dhx;
        };

        // サンプル数
        double N = 50;

        Eigen::Matrix<double, 1, 1> B;
        B << 1.0;
        Eigen::Matrix<double, 1, 1> Q;
        Q << 1.0;
        Eigen::Matrix<double, 1, 1> R;
        R << 100.0;
        Eigen::Matrix<double, 1, 1> P;
        P << 1;

        // システムノイズ
        vector<double> v;
        {
            v.push_back(0.5377);
            v.push_back(1.8339);
            v.push_back(-2.2588);
            v.push_back(0.8622);
            v.push_back(0.3188);
            v.push_back(-1.3077);
            v.push_back(-0.4336);
            v.push_back(0.3426);
            v.push_back(3.5784);
            v.push_back(2.7694);
            v.push_back(-1.3499);
            v.push_back(3.0349);
            v.push_back(0.7254);
            v.push_back(-0.0631);
            v.push_back(0.7147);
            v.push_back(-0.2050);
            v.push_back(-0.1241);
            v.push_back(1.4897);
            v.push_back(1.4090);
            v.push_back(1.4172);
            v.push_back(0.6715);
            v.push_back(-1.2075);
            v.push_back(0.7172);
            v.push_back(1.6302);
            v.push_back(0.4889);
            v.push_back(1.0347);
            v.push_back(0.7269);
            v.push_back(-0.3034);
            v.push_back(0.2939);
            v.push_back(-0.7873);
            v.push_back(0.8884);
            v.push_back(-1.1471);
            v.push_back(-1.0689);
            v.push_back(-0.8095);
            v.push_back(-2.9443);
            v.push_back(1.4384);
            v.push_back(0.3252);
            v.push_back(-0.7549);
            v.push_back(1.3703);
            v.push_back(-1.7115);
            v.push_back(-0.1022);
            v.push_back(-0.2414);
            v.push_back(0.3192);
            v.push_back(0.3129);
            v.push_back(-0.8649);
            v.push_back(-0.0301);
            v.push_back(-0.1649);
            v.push_back(0.6277);
            v.push_back(1.0933);
            v.push_back(1.1093);
        }
        // 観測ノイズ
        vector<double> w;
        {
            w.push_back(-8.6365);
            w.push_back(0.7736);
            w.push_back(-12.1412);
            w.push_back(-11.1350);
            w.push_back(-0.0685);
            w.push_back(15.3263);
            w.push_back(-7.6967);
            w.push_back(3.7138);
            w.push_back(-2.2558);
            w.push_back(11.1736);
            w.push_back(-10.8906);
            w.push_back(0.3256);
            w.push_back(5.5253);
            w.push_back(11.0061);
            w.push_back(15.4421);
            w.push_back(0.8593);
            w.push_back(-14.9159);
            w.push_back(-7.4230);
            w.push_back(-10.6158);
            w.push_back(23.5046);
            w.push_back(-6.1560);
            w.push_back(7.4808);
            w.push_back(-1.9242);
            w.push_back(8.8861);
            w.push_back(-7.6485);
            w.push_back(-14.0227);
            w.push_back(-14.2238);
            w.push_back(4.8819);
            w.push_back(-1.7738);
            w.push_back(-1.9605);
            w.push_back(14.1931);
            w.push_back(2.9158);
            w.push_back(1.9781);
            w.push_back(15.8770);
            w.push_back(-8.0447);
            w.push_back(6.9662);
            w.push_back(8.3509);
            w.push_back(-2.4372);
            w.push_back(2.1567);
            w.push_back(-11.6584);
            w.push_back(-11.4795);
            w.push_back(1.0487);
            w.push_back(7.2225);
            w.push_back(25.8549);
            w.push_back(-6.6689);
            w.push_back(1.8733);
            w.push_back(-0.8249);
            w.push_back(-19.3302);
            w.push_back(-4.3897);
            w.push_back(-17.9468);
        }

        // 真値の初期値
        vector<Eigen::Matrix<double, 1, 1>> x(N);
        x[0] << 10.0;
        //観測値の初期値
        vector<Eigen::Matrix<double, 1, 1>> y(N);
        y[0] << h(x[0]);

        // 真値と観測値を準備
        for (size_t k = 1; k < N; ++k) {
            x[k] << f(x[k-1])(0) + v[k-1];
            y[k] << h(x[k])(0) + w[k];
        }

        // 推定値の初期値
        vector<Eigen::Matrix<double, 1, 1>> x_ = x;
        x_[0] << 10.0 + 1;

        ekf.SetStateSpaceModelFunction(f);
        ekf.SetStateSpaceModelCoefficientJacobian(df);
        ekf.SetObservationFunction(h);
        ekf.SetObservationFunctionJacobian(dh);
        ekf.SetSystemMatrix(B);
        ekf.SetSystemNoiseMatrix(Q);
        ekf.SetObservationNoiseMatrix(R);
        ekf.SetInitialStateMatrix(x_[0]);
        ekf.SetInitialKyobunsanMatrix(P);

        for (int i = 1; i < N; ++i) {
            x_[i] = ekf.Update(y[i]);
        }

        FILE *gplot;
        gplot = popen("gnuplot -persist","w");
        fprintf(gplot, "plot '-' with lines, '-' with lines\n");
        for (int i = 0; i < N; ++i) {
            fprintf(gplot,"%d\t%f\n", i, x[i](0));
        }
        fprintf(gplot,"e\n");
        for (int i = 0; i < N; ++i) {
            fprintf(gplot,"%d\t%f\n", i, x_[i](0));
        }
        fprintf(gplot,"e\n");
        fflush(gplot);
        pclose(gplot);
    }

    {
        ExtendedKalmanFilter ekf;

        Eigen::Matrix<double, 1, 1> x;
        x << 5;

        std::function<Eigen::MatrixXd(Eigen::MatrixXd)> f = [&](Eigen::MatrixXd x) -> Eigen::MatrixXd {
            Eigen::MatrixXd fx = Eigen::MatrixXd::Zero(1, 1);
            fx(0) = x(0) + 5*std::sin(0.1*x(0))*std::sin(0.1*x(0));
            return fx;
        };
        std::function<Eigen::MatrixXd(Eigen::MatrixXd)> df = [&](Eigen::MatrixXd x) -> Eigen::MatrixXd {
            Eigen::MatrixXd dfx = Eigen::MatrixXd::Zero(1, 1);
            dfx(0) = 1 + 5.0*2.0*0.1*cos(0.1*x(0));
            return dfx;
        };
        std::function<Eigen::MatrixXd(Eigen::MatrixXd)> h = [&](Eigen::MatrixXd x) -> Eigen::MatrixXd {
            Eigen::MatrixXd hx = Eigen::MatrixXd::Zero(1, 1);
            hx(0) = x(0);
            return hx;
        };
        std::function<Eigen::MatrixXd(Eigen::MatrixXd)> dh = [&](Eigen::MatrixXd x) -> Eigen::MatrixXd {
            Eigen::MatrixXd dhx = Eigen::MatrixXd::Zero(1, 1);
            dhx(0) = 1;
            return dhx;
        };

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

        std::function<Eigen::MatrixXd(Eigen::MatrixXd)> f = [&](Eigen::MatrixXd x) -> Eigen::MatrixXd {
            Eigen::MatrixXd fx = Eigen::MatrixXd::Zero(3, 1);
            fx(0) = x(0)+T*x(1);
            fx(1) = x(1)+T*(0.5*rho*exp(-x(0)/eta)*x(1)*x(1)*x(2)-g);
            fx(2) = x(2);
            return fx;
        };
        std::function<Eigen::MatrixXd(Eigen::MatrixXd)> h = [&](Eigen::MatrixXd x) -> Eigen::MatrixXd {
            Eigen::MatrixXd hx = Eigen::MatrixXd::Zero(1, 1);
            hx(0) = sqrt(M*M+(x(0)-a)*(x(0)-a));
            return hx;
        };

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
        y[0] << h(x[0])(0) + w[0];

        for (size_t k = 1; k < N; ++k) {
            x[k] = f(x[k-1]);
            y[k] << h(x[k])(0) + w[k];
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