mkdir build
cd build
cmake ..
make

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib:$LD_LIBRARY_PATH
./raytrace_embree_pcl

