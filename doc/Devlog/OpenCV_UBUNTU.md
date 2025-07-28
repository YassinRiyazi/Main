## Installing OpenCV in ubuntu
0. !sudo apt update

1. Ensure you have Nvidia driver
2. CUDA toolkit
3. cuDNN ?

4. OpenGL Dependencies
    !sudo apt install libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev

5. Other Dependencies
    
    !sudo apt install -y build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb-dev libdc1394-dev

    ?libtbb2

6. Git Clone
    1. !git clone https://github.com/opencv/opencv_contrib.git
    2. !cd opencv_contrib
    3. !git checkout 4.10.0
    4. !cd ..

    5. !git clone https://github.com/opencv/opencv.git
    6. !cd opencv
    7. !git checkout 4.10.0
    8. mkdir build
    9. cd build

7. CMake
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
