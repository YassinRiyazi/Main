In case of an error delete and start again.

-----------------------------------------------------------------------------------------------------------------------------------
https://www.jamesbowley.co.uk/qmd/opencv_cuda_python_windows.html
-----------------------------------------------------------------------------------------------------------------------------------
0. Install cuda tool kit 12.8
    CuDNN 9.1.

    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64

    py -m pip install --upgrade pip wheel
    py -m pip install nvidia-cudnn-cu12



1. SO I have added "#include <chrono>" inside two gapi_sample_pipelines.cpp files
2. pip uninstall opencv-python opencv-contrib-python -y


3. Open CMD as Admin
Remember python -c "import numpy; print(numpy.get_include())"
C:\Users\YSN-F\AppData\Roaming\Python\Python312\site-packages\numpy\_core\include

  "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" 

  set CMAKE_BUILD_PARALLEL_LEVEL=16

  "C:\Program Files\CMake\bin\cmake.exe" ^
      -S "C:\Users\YSN-F\Desktop\opencv-4.10.0" ^
      -B "C:\Users\YSN-F\Desktop\OCV_GPU" ^
      -G "Ninja Multi-Config" ^
      -DOPENCV_EXTRA_MODULES_PATH="C:/Users/YSN-F/Desktop/opencv_contrib-4.10.0/modules" ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_INSTALL_PREFIX="C:/Users/YSN-F/Desktop/OCV_GPU/install" ^

      -DWITH_CUDA=ON ^
      -DWITH_OPENGL=ON ^
      -DWITH_OPENCL=ON ^
      -DENABLE_FAST_MATH=ON ^
      -DCUDA_FAST_MATH=ON ^
      -DCUDA_ARCH_BIN=Auto ^
      -DCUDA_GENERATION=Auto ^

      -DBUILD_opencv_world=ON ^
      -DBUILD_opencv_python3=ON ^
      -DINSTALL_PYTHON_EXAMPLES=ON ^
      -DINSTALL_C_EXAMPLES=OFF ^
      -DBUILD_EXAMPLES=ON ^
      -DINSTALL_TESTS=OFF ^

      -DPYTHON3_EXECUTABLE="C:/Program Files/Python312/python.exe" ^
      -DPYTHON3_INCLUDE_DIR="C:/Program Files/Python312/include" ^
      -DPYTHON3_LIBRARY="C:/Program Files/Python312/libs/python312.lib" ^
      -DPYTHON3_PACKAGES_PATH="C:/Program Files/Python312/Lib/site-packages" ^
      -DPYTHON3_NUMPY_INCLUDE_DIRS="C:/Users/YSN-F/AppData/Roaming/Python/Python312/site-packages/numpy/_core/include" ^
      -DOPENCV_ENABLE_NONFREE=ON ^
      -DWITH_QT=OFF


  "C:\Program Files\CMake\bin\cmake.exe" --build "C:/Users/YSN-F/Desktop/OCV_GPU" --target install --config Release


  cmake --build C:/OpenCV --config Release
  cmake --install C:/OpenCV --config Release


4. Add "compile_path\bin\Release" to PATH
  "compile_path\install\x64\vc17\bin"


  python -c "import cv2; print(cv2.__file__); print(cv2.getBuildInformation())"






  "C:\Program Files\CMake\bin\cmake.exe" ^
      -S "C:\Users\robotadmin\Desktop\opencv-4.10.0" ^
      -B "C:\OpenCV" ^
      -G "Ninja Multi-Config" ^
      -DOPENCV_EXTRA_MODULES_PATH="C:/Users/robotadmin/Desktop/opencv_contrib-4.10.0/modules" ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_INSTALL_PREFIX="C:\OpenCV\install" ^

      -DWITH_CUDA=ON ^
      -DWITH_OPENGL=ON ^
      -DWITH_OPENCL=ON ^
      -DENABLE_FAST_MATH=ON ^
      -DCUDA_FAST_MATH=ON ^
      -DCUDA_ARCH_BIN=Auto ^
      -DCUDA_GENERATION=Auto ^

      -DBUILD_opencv_world=ON ^
      -DBUILD_opencv_python3=ON ^
      -DINSTALL_PYTHON_EXAMPLES=ON ^
      -DINSTALL_C_EXAMPLES=OFF ^
      -DBUILD_EXAMPLES=ON ^
      -DINSTALL_TESTS=OFF ^

      -DPYTHON3_EXECUTABLE="C:/Program Files/Python311/python.exe" ^
      -DPYTHON3_INCLUDE_DIR="C:/Program Files/Python311/include" ^
      -DPYTHON3_LIBRARY="C:/Program Files/Python311/libs/python311.lib" ^
      -DPYTHON3_PACKAGES_PATH="C:/Program Files/Python311/Lib/site-packages" ^
      -DPYTHON3_NUMPY_INCLUDE_DIRS="C:/Users/robotadmin/AppData/Roaming/Python/Python311/site-packages/numpy/_core/include" ^
      -DOPENCV_ENABLE_NONFREE=ON ^
      -DWITH_QT=OFF






