// File: show_mesh.cpp

#include <pcl/io/ply_io.h>
#include <pcl/PolygonMesh.h>
#include <pcl/visualization/pcl_visualizer.h>

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <mesh_file.ply>\n";
        return 1;
    }

    std::string mesh_file = argv[1];
    pcl::PolygonMesh mesh;

    // Load mesh from .ply file
    if (pcl::io::loadPLYFile(mesh_file, mesh) < 0) {
        std::cerr << "Error loading mesh file: " << mesh_file << "\n";
        return -1;
    }

    // Visualizer setup
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Mesh Viewer"));
    viewer->setBackgroundColor(0.1, 0.1, 0.1);
    viewer->addPolygonMesh(mesh, "mesh");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }

    return 0;
}
