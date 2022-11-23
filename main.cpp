#include <iostream>
#include "ClassExtraction/extractiondata.h"
#include "Regresion/regresion.h"
#include <eigen3/Eigen/Dense>
#include <list>
#include <vector>
#include <fstream>
#include <boost/algorithm/string.hpp>

int main(int argc, char* argv[])
{
    std::cout<<"RESULTADOS EN C++"<<std::endl;
    //Objeto de ClassExtraction
    ExtractionData ExData(argv[1],argv[2],argv[3]);

    //Se instancia la clase regresion en un objeto
    Regresion modeloRL;

    //Se crea el vector de vectores para cargar objeto ExDara lectura
    std::vector<std::vector<std::string>> dataframe = ExData.LeerCSV();


    //Cantidad de filas y columnas
    int filas = dataframe.size();
    int columnas = dataframe[0].size();

    Eigen::MatrixXd matData = ExData.CSVtoEigen(dataframe, filas, columnas);

    //std::cout<<"filas "<<matData.rows()<<std::endl;
    //std::cout<<"columnas "<<matData.cols()<<std::endl;

    //Se normaliza la matriz de datos

    Eigen::MatrixXd matnorm = ExData.Norm(matData);

    /*std::cout<<"\nmatriz normalizada"<<std::endl;
    for(int i=10033;i<10038;i++)
    {
        std::cout<<matnorm.row(i).col(0)<<" "<<matnorm.row(i).col(1)<<" "<<matnorm.row(i).col(2)<<"..."<<matnorm.row(i).col(28)<<" "<<matnorm.row(i).col(29)<<" "<<matnorm.row(i).col(30)<<std::endl;
    }*/


    //Se divide en datos de entrenamiento y datos de prueba
    Eigen::MatrixXd x_train, y_train, x_test, y_test;

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd , Eigen::MatrixXd> div_datos = ExData.TrainTestSplit(matnorm,0.8);

    //Se descomprime la tupla en 4 conjuntos

    std::tie(x_train, y_train, x_test, y_test) = div_datos;

    /*std::cout <<"\nconjunto de entrenamiento" <<std::endl;
    std::cout <<"\ncolumnas X_train: "<< x_train.cols() <<std::endl;
    std::cout <<"filas X_train: "<< x_train.rows() <<std::endl;
    std::cout <<"columnas Y_train: "<< y_train.cols() <<std::endl;
    std::cout <<"filas Y_train: "<< y_train.rows() <<std::endl;

    std::cout <<"\nconjunto de test" <<std::endl;
    std::cout <<"\ncolumnas  X_test: "<< x_test.cols() <<std::endl;
    std::cout <<"filas X_test: "<< x_test.rows() <<std::endl;
    std::cout <<"columnas Y_test: "<< y_test.cols() <<std::endl;
    std::cout <<"filas Y_test: "<< y_test.rows() <<std::endl;*/


    //Se crea vectores auxiliares para prueba y entrenamiento inicializados en 1
    Eigen::VectorXd V_train = Eigen::VectorXd::Ones(x_train.rows());
    Eigen::VectorXd V_test = Eigen::VectorXd::Ones(x_test.rows());
    /*Se redimenciona el vector de entrenamiento y de prue flotantes para almacenar los valores del costo
    std::vector<float> costo;ba para ser ajustadas
      a los vectores auxiliares anteriores*/


    x_train.conservativeResize(x_train.rows(),x_train.cols()+1);
    x_train.col(x_train.cols()-1) = V_train;

    x_test.conservativeResize(x_test.rows(),x_test.cols()+1);
    x_test.col(x_test.cols()-1) = V_test;

    //Se crea el vector de coeficientes theta
    Eigen::VectorXd thetas= Eigen::VectorXd::Zero(x_train.cols());

    //Se establece el alpha como ratio de aprendizaje
    float alpha = 0.01;
    int num_iter = 1000;

    //Se crea un vector para almacenar las thetas de salida (m y b)
    Eigen::MatrixXd thetas_out;

    //Se crea un vector sencillo(std) de flotantes para almacenar los valores del costo
    std::vector<float> costo;

    //Se calcula el gradiente decendiente
    std::tuple<Eigen::VectorXd, std::vector<float>> g_decendiente = modeloRL.GradientDescent(x_train,y_train,thetas,alpha,num_iter);

    //Se desempaqueta el gradiente
    std::tie(thetas_out,costo) = g_decendiente;
    //std::cout <<"thetas_out\n "<<thetas_out<<std::endl;


    //Se almacena los valores de tethas y costos en un fichero para posteriormente ser visualizados
    //ExData.VectorToFile(costo,"costos.txt");
    //ExData.EigenToFile(thetas_out,"thetas.txt");

    //Se extrae el promedio de la matriz de entrada
    auto prom_data = ExData.Promedio(matData);// ···········································································matData
    //std::cout<<"Promedio matriz de entrada: \n"<<prom_data<<std::endl;


    //Se extraen los valores de las variables independientes
    auto var_prom_independiente = prom_data(0,30);// ···········································································11

    //std::cout<<"var_prom_independiente: \n"<<var_prom_independiente<<std::endl;

    //Se extraen los datos
    auto datos_escalados = matData.rowwise()-matData.colwise().mean();

    //std::cout<<"datos_escalados \n"<<datos_escalados<<std::endl;

    //Se extrae la desviacion estandar de los datos escalados
    auto desv_stand = ExData.DevStand(datos_escalados);
    std::cout<<"desviacion estandar \n"<<desv_stand<<std::endl;


    //Se extraen los valores de la variables independientes
    auto var_desv_independientes = desv_stand(0,30);// ···········································································11


    //Se crea una matriz para almacenar lo valores estimados de entrenamiento
    Eigen::MatrixXd y_train_hat = (x_train*thetas_out*var_desv_independientes).array()+var_prom_independiente;
    //std::cout<<"y_train_hat \n"<<y_train_hat<<std::endl;

    //Matriz para los valores reales de y (80% de los datos)
    Eigen::MatrixXd y = matData.col(30).topRows(8030);// ···········································································11-1278

    //Se revisa que tan bueno fue el modelo a traves de la metrica de rendimiento
    float metrica_R2 = modeloRL.R2_Score(y,y_train_hat);
    std::cout<<"Metrica R2 entrenamiento: "<<metrica_R2<<std::endl;

    /*Eigen::MatrixXd y_test_hat = (x_test*thetas_out*var_desv_independientes).array()+var_prom_independiente;
    Eigen::MatrixXd ytest = matData.col(11).bottomRows(320);
    float metrica_R2_test = modeloRL.R2_Score(ytest,y_test_hat);

    std::cout<<"Metrica R2 prueba: "<<metrica_R2_test<<std::endl;*/

    return EXIT_SUCCESS;
}
