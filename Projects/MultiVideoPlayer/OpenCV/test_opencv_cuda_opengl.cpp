#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

int main() {
    // Test CUDA support
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;

    try {
        int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
        if (cuda_devices > 0) {
            std::cout << "CUDA Support: YES" << std::endl;
            std::cout << "Number of CUDA devices: " << cuda_devices << std::endl;
            cv::cuda::DeviceInfo device_info;
            std::cout << "CUDA Device Name: " << device_info.name() << std::endl;
        } else {
            std::cout << "CUDA Support: NO (No CUDA-enabled devices found)" << std::endl;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "CUDA Support: NO (Error: " << e.what() << ")" << std::endl;
    }

    // Test OpenGL support
    try {
        cv::namedWindow("OpenGL Test Window", cv::WINDOW_OPENGL);
        std::cout << "OpenGL Support: YES" << std::endl;
        cv::destroyWindow("OpenGL Test Window");
    } catch (const cv::Exception& e) {
        std::cerr << "OpenGL Support: NO (Error: " << e.what() << ")" << std::endl;
    }

    return 0;
}


/*

export LD_LIBRARY_PATH=/home/d2u25/OCV_GPU/install/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH


g++ -o test_opencv_cuda_opengl test_opencv_cuda_opengl.cpp \
-I/home/d2u25/OCV_GPU/install/include/opencv4 \
-L/home/d2u25/OCV_GPU/install/lib \
-lopencv_core -lopencv_highgui -lopencv_cudaarithm \
-std=c++11

*/