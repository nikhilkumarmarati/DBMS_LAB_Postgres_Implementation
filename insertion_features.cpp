#include <iostream>
#include "feature_extractor.h"
#include "stl10_loader.h"
#include <sql.h>
#include <sqlext.h>

using namespace std;

typedef SQLHENV HENV;
typedef SQLHDBC HDBC;
typedef SQLHSTMT HSTMT;
typedef SQLRETURN RET;

const string FEATURE_FILE = "features.txt";

// Function to extract features from images
std::unordered_map<int, std::vector<double>> extractFeatures(const std::vector<cv::Mat>& images, FeatureExtractor& extractor) {
    std::unordered_map<int, std::vector<double>> features;
    ofstream file(FEATURE_FILE);  // Open file for writing

    for (size_t i = 0; i < images.size(); i++) {
        std::vector<double> feature = extractor.extract(images[i]);
        features[i] = feature;

        // Save to file
        file << i;
        for (double val : feature) {
            file << " " << val;
        }
        file << "\n";

        if ((i + 1) % 100 == 0 || i == images.size() - 1) {
            std::cout << "Extracted features for " << (i + 1) << " / " << images.size() << " images" << std::endl;
        }
    }

    file.close();
    return features;
}

// Function to load features from a file
std::unordered_map<int, std::vector<double>> loadFeaturesFromFile(const string& filename) {
    std::unordered_map<int, std::vector<double>> features;
    ifstream file(filename);

    if (!file) {
        cerr << "Error: Unable to open " << filename << endl;
        return {};
    }

    int image_id;
    while (file >> image_id) {
        std::vector<double> feature_vector(25);  // Assuming 25 features
        for (double& val : feature_vector) {
            file >> val;
        }
        features[image_id] = feature_vector;
    }

    file.close();
    return features;
}

// Function to execute SQL queries
SQLHSTMT executeSQL(HDBC hdbc, const string& query) {
    SQLHSTMT hstmt;
    SQLRETURN ret;

    // Allocate statement handle
    ret = SQLAllocHandle(SQL_HANDLE_STMT, hdbc, &hstmt);
    if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
        cerr << "Error allocating statement handle" << endl;
        return NULL;
    }

    // Execute the SQL query
    ret = SQLExecDirect(hstmt, (SQLCHAR*)query.c_str(), SQL_NTS);
    if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
        cerr << "Error executing query: " << query << endl;
        SQLFreeHandle(SQL_HANDLE_STMT, hstmt);
        return NULL;
    }

    return hstmt; // Return the statement handle to be used for fetching data
}

int main() {
    HENV henv;
    HDBC hdbc;
    RET ret;
    HSTMT result;
    SQLCHAR row[256];

    // Allocate environment handle
    ret = SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &henv);
    if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
        cerr << "Error allocating environment handle" << endl;
        return -1;
    }

    // Set the ODBC version environment attribute
    ret = SQLSetEnvAttr(henv, SQL_ATTR_ODBC_VERSION, (SQLPOINTER)SQL_OV_ODBC3, 0);
    if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
        cerr << "Error setting ODBC version" << endl;
        return -1;
    }
    
    // Allocate connection handle
    ret = SQLAllocHandle(SQL_HANDLE_DBC, henv, &hdbc);
    if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
        cerr << "Error allocating connection handle" << endl;
        return -1;
    }

    // Connect to the database
    // SQLCHAR connectionString[] = "Driver={PostgreSQL Unicode};Server=10.5.18.70;Database=22CS10042;Uid=22CS10042;Pwd=22CS10042;";
    SQLCHAR connectionString[] = "Driver={PostgreSQL Unicode};Server=172.24.192.1;Database=postgres;Uid=postgres;Pwd=Nn9959579946;";
    ret = SQLDriverConnect(hdbc, NULL, connectionString, SQL_NTS, NULL, 0, NULL, SQL_DRIVER_COMPLETE);

    if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
        cerr << "Error connecting to database" << endl;
        SQLCHAR sqlState[6], message[256];
        SQLINTEGER nativeError;
        SQLSMALLINT messageLength;
        SQLGetDiagRec(SQL_HANDLE_DBC, hdbc, 1, sqlState, &nativeError, message, sizeof(message), &messageLength);
        cerr << "SQLState: " << sqlState << ", Message: " << message << endl;

        return -1;
    }

    printf("Connected to the database successfully!\n");

     // Load features if available, otherwise extract
     std::unordered_map<int, std::vector<double>> features;
     ifstream featureFile(FEATURE_FILE);
     if (featureFile.good()) {
         cout << "Loading features from file..." << endl;
         features = loadFeaturesFromFile(FEATURE_FILE);
     } else {
         cout << "Extracting features from images..." << endl;
         STL10Loader loader("stl10_binary");
         std::vector<cv::Mat> images = loader.loadImages("train");
 
         FeatureExtractor extractor("dinov2_pca_25d.pt");
         features = extractFeatures(images, extractor);
     }

    //insert features into the database
    for (const auto& pair : features) {
        int image_id = pair.first;
        const std::vector<double>& feature_vector = pair.second;
    
        // Check if we have the expected number of features
        if (feature_vector.size() != 25) {
            cerr << "Warning: Image ID " << image_id << " has " << feature_vector.size() 
                 << " features instead of 25" << endl;
            continue;
        }
        
        // Prepare SQL statement to insert features into the database
        std::string sql = "INSERT INTO image (id, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, "
                          "f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, "
                          "f21, f22, f23, f24, f25) VALUES (" + std::to_string(image_id) + ", ";
        
        for (size_t i = 0; i < feature_vector.size(); ++i) {
            sql += std::to_string(feature_vector[i]);
            if (i < feature_vector.size() - 1) {
                sql += ", ";
            }
        }
        sql += ")";
        
        
        // Execute the SQL statement
        result = executeSQL(hdbc, sql);
        if (result == NULL) {
            cerr << "Error inserting features for image ID: " << image_id << endl;
            continue;
        }
    }

    cout << "Features inserted into the database successfully!" << endl;

}