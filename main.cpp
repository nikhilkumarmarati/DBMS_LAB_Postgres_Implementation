#include "query_image.h"
#include "stl10_loader.h"
#include "feature_extractor.h"


// Function to extract features from images
std::unordered_map<int, std::vector<double>> extractFeatures(const std::vector<cv::Mat>& images, FeatureExtractor& extractor) {
    std::unordered_map<int, std::vector<double>> features;
    
    for (size_t i = 0; i < images.size(); i++) {
        std::vector<double> feature = extractor.extract(images[i]);
        features[i] = feature;
        
        if ((i + 1) % 100 == 0 || i == images.size() - 1) {
            std::cout << "Extracted features for " << (i + 1) << " / " << images.size() << " images" << std::endl;
        }
    }
    
    return features;
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
        return -1;
    }

    printf("Connected to the database successfully!\n");

    // Load STL-10 dataset for queries
    STL10Loader loader("stl10_binary");
    std::vector<cv::Mat> query_images = loader.loadImages("test", 100);
    
    // Initialize feature extractor
    FeatureExtractor extractor("dinov2_pca_25d.pt");
    
    // Extract features for query images
    auto query_features = extractFeatures(query_images, extractor);

    //start the timer
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int count = 1;
    // Perform k-NN queries
    for (const auto& [id, query_point] : query_features) {
        std::cout << "Query image " << count++ << ":" << std::endl;
        auto results = query_image(query_point, hdbc,5);
        
        std::cout << "Top 5 nearest neighbors:" << std::endl;
        for (const auto& [result_id, dist] : results) {
            std::cout << "Image ID: " << result_id << ", Distance: " << dist << std::endl;
        }
        std::cout << std::endl;
    }
    
    //stop the timer
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Total query time: " << duration.count() << " ms" << std::endl;
    std::cout << "Average query time: " << (duration.count() / query_features.size()) << " ms" << std::endl;


}