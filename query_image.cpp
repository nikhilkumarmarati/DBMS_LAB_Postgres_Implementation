/*
c++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) query_image.cpp -o query_image.so $(python3-config --ldflags) -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lodbc -ltorch -ltorch_cpu -lc10
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "feature_extractor.h"  // Include the feature extractor header
#include "query_image.h"        // Include the image query header
#include <sql.h>
#include <sqlext.h>

typedef SQLHENV HENV;
typedef SQLHDBC HDBC;
typedef SQLHSTMT HSTMT;
typedef SQLRETURN RET;

namespace py = pybind11;

// Wrapper for FeatureExtractor class
class PyFeatureExtractor {
private:
    FeatureExtractor extractor;

public:
    PyFeatureExtractor(const std::string& model_path) : extractor(model_path) {}
    
    // Method to extract features from numpy array
    std::vector<double> extract(py::array_t<unsigned char> input_array) {
        // Get info about the input array
        py::buffer_info buffer = input_array.request();
        
        // Handle different dimensions (grayscale vs color)
        cv::Mat image;
        if (buffer.ndim == 2) {
            // Grayscale image
            image = cv::Mat(buffer.shape[0], buffer.shape[1], CV_8UC1, buffer.ptr);
        } 
        else if (buffer.ndim == 3 && buffer.shape[2] == 3) {
            // Color image
            image = cv::Mat(buffer.shape[0], buffer.shape[1], CV_8UC3, buffer.ptr);
        }
        else {
            throw std::runtime_error("Input array must be a 2D grayscale image or 3D color image with 3 channels");
        }
        
        // Use the actual extract method from feature_extractor.h
        return extractor.extract(image);
    }
    
    // Expose the feature dimension
    int getFeatureDim() const {
        return extractor.getFeatureDim();
    }
};

// Wrapper for database connection to be used from Python
class DatabaseConnection {
private:
    SQLHDBC henv;
    HDBC hdbc;
    HSTMT hstmt;
    SQLRETURN ret;
    bool connected;

public:
    DatabaseConnection() : connected(false) {}
    
    bool connect(const std::string& dsn, const std::string& username, const std::string& password) {

        SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &henv);
        SQLSetEnvAttr(henv, SQL_ATTR_ODBC_VERSION, (void*)SQL_OV_ODBC3, 0);
        SQLAllocHandle(SQL_HANDLE_DBC, henv, &hdbc);
    
        // SQLCHAR connectionString[] = "Driver={PostgreSQL Unicode};Server=10.5.18.70;Database=22CS10042;Uid=22CS10042;Pwd=22CS10042;";
    SQLCHAR connectionString[] = "Driver={PostgreSQL Unicode};Server=172.24.192.1;Database=postgres;Uid=postgres;Pwd=Nn9959579946;";
        ret = SQLDriverConnect(hdbc, NULL, connectionString, SQL_NTS, NULL, 0, NULL, SQL_DRIVER_COMPLETE);
        
        if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
            std::cerr << "Error connecting to database" << std::endl;
            return false;
        }
    
        connected = true;
        return true;
    }
    
    
    // Wrapper for the query_image function
    std::vector<std::pair<int, double>> query(const std::vector<double>& features, int k = 5) {
        if (!connected) {
            throw std::runtime_error("Database not connected");
        }
        return query_image(features, hdbc, k);
    }
    
    void disconnect() {
        if (connected) {
            SQLDisconnect(hdbc);
            SQLFreeHandle(SQL_HANDLE_DBC, hdbc);
            SQLFreeHandle(SQL_HANDLE_ENV, henv);
            connected = false;
        }
    }
    
    ~DatabaseConnection() {
        disconnect();
    }
};

// Expose functionality to Python
PYBIND11_MODULE(query_image, m) {
    m.doc() = "Image feature extraction and query module";
    
    // Expose FeatureExtractor class instead of just a function
    py::class_<PyFeatureExtractor>(m, "FeatureExtractor")
        .def(py::init<const std::string&>())
        .def("extract", &PyFeatureExtractor::extract,
             "Extract features from an image")
        .def("getFeatureDim", &PyFeatureExtractor::getFeatureDim,
             "Get the dimension of the feature vector");
    
    // Expose database connection class
    py::class_<DatabaseConnection>(m, "DatabaseConnection")
        .def(py::init<>())
        .def("connect", &DatabaseConnection::connect, 
             "Connect to database", 
             py::arg("dsn"), py::arg("username"), py::arg("password"))
        .def("query", &DatabaseConnection::query, 
             "Query the database with feature vector", 
             py::arg("features"), py::arg("k") = 5)
        .def("disconnect", &DatabaseConnection::disconnect, 
             "Disconnect from database");
}