#ifndef REGRESION_H
#define REGRESION_H
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <cmath>


class Regresion
{
public:
   float F_OLS_Cost(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd thetas);
   std::tuple<Eigen::VectorXd, std::vector<float>> GradientDescent(Eigen::MatrixXd X,
                                                                              Eigen::MatrixXd y,
                                                                              Eigen::VectorXd thetas,
                                                                              float alpha,
                                                                              int num_iter);
   float R2_Score(Eigen::MatrixXd y, Eigen::MatrixXd y_hat);
   float MSE(Eigen::MatrixXd y, Eigen::MatrixXd y_hat);
   float RMSE(Eigen::MatrixXd y, Eigen::MatrixXd y_hat);
};

#endif // REGRESION_H
