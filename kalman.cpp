#include <iostream>
#include <cmath>
#include <random>
#include "kalmanfilter.hpp"

using namespace std;
using namespace Eigen;



int main() {
    KalmanFilter kf;
    kf.Init();

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
    return 0;
}