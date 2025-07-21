g++ -std=c++14 main.cpp -o show_mesh \
    $(pkg-config --cflags --libs pcl_io pcl_visualization) \
    -I/usr/include/vtk-9.3 \
    -lvtkCommonCore-9.3 -lvtkRenderingCore-9.3 -lvtkIOGeometry-9.3 \
    -lvtkFiltersCore-9.3 -lvtkInteractionStyle-9.3 -lvtkRenderingOpenGL2-9.3 \
    -lvtksys-9.3
