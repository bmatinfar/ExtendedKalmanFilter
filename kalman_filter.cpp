#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    MatrixXd I = MatrixXd::Identity(4, 4);
    VectorXd y = z - H_ * x_;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd K =  P_ * Ht * Si;
    
    x_ = x_ + K * y;
    P_ = (I - K*H_)* P_.transpose();
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    MatrixXd I = MatrixXd::Identity(4, 4);

    float x = x_(0);
    float y = x_(1);
    float vx = x_(2);
    float vy = x_(3);
    
    float rho = sqrtf(x*x + y*y);
    float theta = atan2f(y, x);
    if(rho < 1e-5)
        rho = 1e-5;
    float rho_dot = (x*vx + y*vy)/rho;
    VectorXd z_pred = VectorXd(3);
    z_pred << rho, theta, rho_dot;
    
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd K =  P_ * Ht * Si;
    VectorXd yz = (z - z_pred);
    
    while(yz(1) < -M_PI)
        yz(1) += 2*M_PI;
    while(yz(1) > M_PI)
        yz(1) -= 2*M_PI;
    
    x_ = x_ + K * yz;
    P_ = (I - K*H_)* P_.transpose();
    
}
