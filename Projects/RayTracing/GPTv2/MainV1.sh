# conda deactivate
g++ -std=c++17 MainV1.cpp -o ./so/show_mesh.so \
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

./so/show_mesh.so ./Poly/hemisphere_mesh_exponential.ply 