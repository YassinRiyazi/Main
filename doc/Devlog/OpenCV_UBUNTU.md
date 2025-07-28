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
    git clone --branch 4.10.0 https://github.com/opencv/opencv.git /home/d2u25/opencv-4.10.0
    git clone --branch 4.10.0 https://github.com/opencv/opencv_contrib.git /home/d2u25/opencv_contrib-4.10.0


7.1. Clear /home/d2u25/OCV_GPU for CMake attempt

7. CMake
Achtung: nvcc is not compatible with gcc 12
    export CC=/usr/bin/gcc-11
    export CXX=/usr/bin/g++-11

  cmake \
  -S /home/d2u25/opencv-4.10.0 \
  -B /home/d2u25/OCV_GPU \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/home/d2u25/OCV_GPU/install \
  -DOPENCV_EXTRA_MODULES_PATH=/home/d2u25/opencv_contrib-4.10.0/modules \
  -DWITH_CUDA=ON \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DENABLE_FAST_MATH=ON \
  -DCUDA_FAST_MATH=ON \
  -DCUDA_ARCH_BIN=7.5;8.6 \
  -DCUDA_ARCH_PTX=8.6 \
  -DCUDA_GENERATION=Auto \
  -DBUILD_opencv_world=ON \
  -DBUILD_opencv_python3=ON \
  -DINSTALL_PYTHON_EXAMPLES=ON \
  -DINSTALL_C_EXAMPLES=OFF \
  -DBUILD_EXAMPLES=ON \
  -DINSTALL_TESTS=OFF \
  -DPYTHON3_EXECUTABLE=/home/d2u25/anaconda3/envs/torch3.11/bin/python \
  -DPYTHON3_INCLUDE_DIR=/home/d2u25/anaconda3/envs/torch3.11/include/python3.11 \
  -DPYTHON3_LIBRARY=/home/d2u25/anaconda3/envs/torch3.11/lib/libpython3.11.so \
  -DPYTHON3_PACKAGES_PATH=/home/d2u25/anaconda3/envs/torch3.11/lib/python3.11/site-packages \
  -DPYTHON3_NUMPY_INCLUDE_DIRS=/home/d2u25/anaconda3/envs/torch3.11/lib/python3.11/site-packages/numpy/_core/include \
  -DOPENCV_ENABLE_NONFREE=ON \
  -DWITH_QT=OFF \
  -DWITH_CUDNN=ON \
  -DOPENCV_DNN_CUDA=ON \
  -DOPENCV_GENERATE_PKGCONFIG=ON \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-11 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF 

Compile OpenCV using all available CPU cores (replace $(nproc) with the number of cores if needed):
8. make -j$(nproc)








Problems:
    -DCUDA_ARCH_BIN=6.0,8.9\ Comma is not accepted

    -DOPENCV_EXTRA_MODULES_PATH= /home/d2u25/opencv_contrib-4.10.0/modules \  Correct it like this (no space after =):
