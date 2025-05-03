/*g++ query_image.cpp -o query_image -lodbc -lodbcinst
*/

#include <iostream>
#include <sql.h>
#include <sqlext.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <utility> // for std::pair
#include <iomanip> // for std::setprecision
#include <sstream> // for std::ostringstream
#include <queue> // for std::priority_queue



using namespace std;

typedef SQLHENV HENV;
typedef SQLHDBC HDBC;
typedef SQLHSTMT HSTMT;
typedef SQLRETURN RET;

std::vector<std::pair<int, double>> load_features_from_db(const HDBC& hdbc, const string& query) {
    SQLHSTMT hstmt;
    SQLRETURN ret;
    std::vector<std::pair<int, double>> result;

    // Allocate statement handle
    ret = SQLAllocHandle(SQL_HANDLE_STMT, hdbc, &hstmt);
    if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
        cerr << "Error allocating statement handle" << endl;
        return result; // Return empty vector on error
    }

    // Execute the SQL query
    ret = SQLExecDirect(hstmt, (SQLCHAR*)query.c_str(), SQL_NTS);
    if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
        std::cerr << "Error executing query: " << query << std::endl;
    
        SQLCHAR sqlState[6];
        SQLINTEGER nativeError;
        SQLCHAR messageText[256];
        SQLSMALLINT textLength;
    
        int i = 1;
        while (SQLGetDiagRec(SQL_HANDLE_STMT, hstmt, i, sqlState, &nativeError,
                             messageText, sizeof(messageText), &textLength) == SQL_SUCCESS) {
            std::cerr << "ODBC Error " << i << ":\n";
            std::cerr << "  SQLSTATE: " << reinterpret_cast<const char*>(sqlState) << "\n";
            std::cerr << "  Native Error: " << nativeError << "\n";
            std::cerr << "  Message: " << reinterpret_cast<const char*>(messageText) << "\n";
            ++i;
        }
    
        SQLFreeHandle(SQL_HANDLE_STMT, hstmt);
        return result; // Return empty vector on error
    }
    

    // Fetch the results
    while (SQLFetch(hstmt) == SQL_SUCCESS) {
        int image_id;
        double dis;  // Assuming 25 features

        //get the image id
        ret = SQLGetData(hstmt, 1, SQL_C_LONG, &image_id, 0, NULL);
        if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
            cerr << "Error fetching image ID" << endl;
            continue; // Skip this row
        }

        ret = SQLGetData(hstmt, 2, SQL_C_DOUBLE, &dis, 0, NULL); // Assuming features start from column 2
        if (ret != SQL_SUCCESS && ret != SQL_SUCCESS_WITH_INFO) {
            cerr << "Error fetching distance" << endl;
            continue;
        }
        result.push_back(std::make_pair(image_id, dis)); // Store the image ID and features in the vector
    }

    // Free statement handle
    SQLFreeHandle(SQL_HANDLE_STMT, hstmt);
    
    return result; // Return the loaded features
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


// Function to find the k nearest neighbors to a query image
std::vector<std::pair<int, double>> query_image(const std::vector<double>& query_feature, HDBC hdbc, int k = 5) {
    // Load features from database
    // Initialize the query string
    std::string s = "SELECT id, ";

    // Add the distance calculation part dynamically using the query_feature values
    for (size_t i = 0; i < query_feature.size(); ++i) {
        if (i > 0) {
            s += " + ";  // Add the '+' operator for subsequent terms
        }
        // Use POWER() instead of ^2
        s += "POWER(f" + std::to_string(i + 1) + " - (" + std::to_string(query_feature[i]) + "), 2)";
    }
    
    // Add the rest of the SQL query
    s += " AS distance FROM image ORDER BY distance ASC LIMIT " + std::to_string(k) + ";";
    

    
    std::vector<std::pair<int, double>> results = load_features_from_db(hdbc, s);

    return results;
}