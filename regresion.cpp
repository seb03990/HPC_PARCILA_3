#include "regresion.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <cmath>

/*Primera funcion:
Funcion de costo para la regreseion lineal
basada en los minimos cuadrados ordinarios
demostrado en clase*/

float Regresion::F_OLS_Cost(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd thetas){

    Eigen::MatrixXd m_interior = (pow((X*thetas - y).array(),2));

    return m_interior.sum()/(2*X.rows());
}

/* Funcion de gradiente descendiente:
   En funcion de un ratio de aprendizaje se avanza hasta encontrar
   el punto minimo que representa el valor optimo para la funcion*/

std::tuple<Eigen::VectorXd, std::vector<float>> Regresion::GradientDescent(Eigen::MatrixXd X,
                                                                           Eigen::MatrixXd y,
                                                                           Eigen::VectorXd thetas,
                                                                           float alpha,
                                                                           int num_iter){
    Eigen::MatrixXd temporal = thetas;
    int parametros = thetas.rows();
    std::vector<float> costo;

    //En costo ingresaremos los valores de la funcion de costo
    costo.push_back(F_OLS_Cost(X,y,thetas));

    /*Se itera segun el numero de iteraciones y el ratio de aprendizaje
      para encontrar los valores optimos*/

    for(int i=0; i<num_iter;i++)
    {
        Eigen::MatrixXd error = X*thetas-y;

        for(int j=0; j<parametros; j++)
        {
            Eigen::MatrixXd X_i = X.col(j);
            Eigen::MatrixXd termino = error.cwiseProduct(X_i);
            temporal(j,0) = thetas(j,0) - (alpha/X.rows())*termino.sum();
        }
        thetas = temporal;
        costo.push_back(F_OLS_Cost(X,y,thetas));
    }

    return std::make_tuple(thetas,costo);
}

/*A continuacion se presenta la funcion para revisar que tan bueno es nuestro modelo:
 *Se procede a crear la metrica de rendimiento:
 *Vamos a crear r cuadrado score:
 *Coeficiente de determinacion, en donde el mejor valor posible es 1
 *Vamos a crear un flotante entre 0 y 1*/

float Regresion::R2_Score(Eigen::MatrixXd y, Eigen::MatrixXd y_hat){
    auto numerador = pow((y-y_hat).array(),2).sum();
    auto denominador = pow((y.array()-y.mean()),2).sum();

    return 1-(numerador/denominador);
}

float Regresion::MSE(Eigen::MatrixXd y, Eigen::MatrixXd y_hat){
    auto numerador = pow((y-y_hat).array(),2).sum();
    return (numerador/y.rows());
}

float Regresion::RMSE(Eigen::MatrixXd y, Eigen::MatrixXd y_hat){
    return sqrt(MSE(y,y_hat));
}


















