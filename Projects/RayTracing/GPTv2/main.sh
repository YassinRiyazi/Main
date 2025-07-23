# conda deactivate
g++ -std=c++17 main.cpp -o show_mesh \
    -I/usr/include/vtk-9.3 \
    $(pkg-config --cflags --libs pcl_io pcl_visualization) \
    -L/usr/lib/x86_64-linux-gnu\
    -lvtkRenderingOpenGL2-9.3 \
    -lvtkInteractionStyle-9.3 \
    -L/usr/lib/x86_64-linux-gnu\
    -lvtkRenderingLOD-9.3 \
    -lvtkIOGeometry-9.3 \
    -lvtkRenderingCore-9.3 \
    -lvtkCommonCore-9.3 \
    -lvtksys-9.3\
    -lvtkFiltersSources-9.3\
    -lvtkCommonExecutionModel-9.3\
    -lvtkCommonDataModel-9.3\
    -lvtkCommonMath-9.3\

./show_mesh ./Poly/hemisphere_mesh_exponential.ply 