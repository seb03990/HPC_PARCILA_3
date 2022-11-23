#include "extractiondata.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <list>
#include <vector>
#include <fstream>
#include <boost/algorithm/string.hpp>

/* Primera funcion miembro:
 * Lectira del fichero csv.
 * Se almacena en un vector de vectores del tipo string*/

std::vector<std::vector<std::string>> ExtractionData::LeerCSV(){
    /* En primer lugar se abre y se almacena el
     * fichero en un buffer temporal o variable emporal "archivo"**/
    std::fstream archivo(dataset);
    // Se crea un vector de vectores del tipo String
    std::vector<std::vector<std::string>> datosString;

    std::string linea="";
    while(getline(archivo,linea)){
        std::vector<std::string> vector;
        /* Se identifica el elemento que compone cada vector
         * Se divide cada elemento con boost
         */

        boost::algorithm::split(vector,linea,boost::is_any_of(delimitador));
        //finalmente se ingresa al buffer temporal
        datosString.push_back(vector);
    }

    //Se cierra el fichero csv
    archivo.close();
    //Se retorna el vector de vectores
    return datosString;
}

Eigen::MatrixXd ExtractionData::CSVtoEigen(std::vector<std::vector<std::string>> dataSet, int filas, int columnas){ 


    if(header)
    {
        filas = filas-1;
    }

    Eigen::MatrixXd matriz(columnas,filas);
    for(int i=0;i<filas;i++){
        for(int j=0;j<columnas;j++){
            matriz(j,i) = atof(dataSet[i][j].c_str());
        }
    }

    return matriz.transpose();
}


//Funcion para extraer el promedio:

/*Cuando el programador no sabe que tipo de dato
  regresara la funcion se una auto "variable" decltype*/

auto ExtractionData::Promedio(Eigen::MatrixXd datos) -> decltype(datos.colwise().mean()){

    return datos.colwise().mean();
}

//Funcion para extraer la desviacion estandar

auto ExtractionData::DevStand(Eigen::MatrixXd datos) -> decltype(((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt()){

    return ((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt();
}

/*Funcion para normalizar los datos

  Se retorna la matriz normalizada
  Se toma como argumentos la matriz de datos*/

Eigen::MatrixXd ExtractionData::Norm(Eigen::MatrixXd datos){

    //Se escalan los datos: Xi-mean
    Eigen::MatrixXd mat_escalado = datos.rowwise()-Promedio(datos);

    //Se calcula la normalizacion
    Eigen::MatrixXd mat_normal = mat_escalado.array().rowwise()/DevStand(mat_escalado);

    return mat_normal;
}

/* Funcion para dividir en 4 grandes grupos los datos:
 * x_train
 * y_train
 * x_test
 * y_test*/

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd , Eigen::MatrixXd> ExtractionData::TrainTestSplit(Eigen::MatrixXd datos, float size_train){
    //Cantidad de filas totales
    int filas_totales = datos.rows();
    //Cantidad de filas para entrenamiento
    int filas_train = round(filas_totales*size_train);
    //Cantidad de filas de prueba
    int filas_test = filas_totales-filas_train;

    Eigen::MatrixXd Train = datos.topRows(filas_train);

    //Se desprende para dependientes e independientes

    Eigen::MatrixXd x_train = Train.leftCols(datos.cols()-1);
    Eigen::MatrixXd y_train = Train.rightCols(1);

    Eigen::MatrixXd Test = datos.bottomRows(filas_test);

    Eigen::MatrixXd x_test = Test.leftCols(datos.cols()-1);
    Eigen::MatrixXd y_test = Test.rightCols(1);

    //Se compacta la tupla y se retorna

    return std::make_tuple(x_train,y_train,x_test,y_test);

}

//Para efectos de visualizacion se creara la funcion vector a fichero

void ExtractionData::VectorToFile(std::vector<float> vector, std::string file_name){
    std::ofstream file_salida(file_name);

    //Se crea el iterador para almacenar la salida del vector
    std::ostream_iterator<float> salida_iterador(file_salida,"\n");

    //Se copia cada valor desde el inicio hasta el fin del iterador en el fichero
    std::copy(vector.begin(),vector.end(),salida_iterador);
}

//Para efectos de manipulacion  visualizacion se crea la funcion matriz Eigeen a fichero

void ExtractionData::EigenToFile(Eigen::MatrixXd matriz, std::string file_name){
    std::ofstream file_salida(file_name);

    if(file_salida.is_open())
    {
        file_salida << matriz <<"\n";
    }
}


































