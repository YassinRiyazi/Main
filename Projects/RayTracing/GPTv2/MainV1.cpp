/*
    Assumptions:
        Substrate is not reflective.
*/

#include <pcl/io/ply_io.h>
#include <pcl/PolygonMesh.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <cmath>
#include <sstream>

// Function to load mesh from a .ply file
int load_mesh(const std::string& mesh_file, pcl::PolygonMesh& mesh)
{
    // Load mesh from .ply file
    if (pcl::io::loadPLYFile(mesh_file, mesh) < 0) {
        std::cerr << "Error loading mesh file: " << mesh_file << "\n";
        return -1;
    }
    return 0;
}

// Function to compute the axis-aligned bounding box (AABB) of the mesh
std::pair<pcl::PointXYZ, pcl::PointXYZ> getMeshBoundary(const pcl::PolygonMesh& mesh)
{
    // Convert mesh.cloud to PointCloud<PointXYZ>
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(mesh.cloud, *cloud);

    // Check if the cloud is empty
    if (cloud->points.empty()) {
        std::cerr << "Error: Mesh has no vertices.\n";
        return std::make_pair(pcl::PointXYZ(0, 0, 0), pcl::PointXYZ(0, 0, 0));
    }

    // Initialize min and max points with the first vertex
    pcl::PointXYZ min_point = cloud->points[0];
    pcl::PointXYZ max_point = cloud->points[0];

    // Iterate through all vertices to find min and max coordinates
    for (const auto& point : cloud->points) {
        min_point.x = std::min(min_point.x, point.x);
        min_point.y = std::min(min_point.y, point.y);
        min_point.z = std::min(min_point.z, point.z);
        max_point.x = std::max(max_point.x, point.x);
        max_point.y = std::max(max_point.y, point.y);
        max_point.z = std::max(max_point.z, point.z);
    }

    return std::make_pair(min_point, max_point);
}

// Function to compute ray-triangle intersection using Möller-Trumbore algorithm
bool rayTriangleIntersection(const pcl::PointXYZ& ray_origin, const pcl::PointXYZ& ray_direction,
                            const pcl::PointXYZ& v0, const pcl::PointXYZ& v1, const pcl::PointXYZ& v2,
                            float& t)
{
    const float EPSILON = 1e-8;

    // Convert PCL points to Eigen vectors for computation
    Eigen::Vector3f origin(ray_origin.x, ray_origin.y, ray_origin.z);
    Eigen::Vector3f direction(ray_direction.x, ray_direction.y, ray_direction.z);
    Eigen::Vector3f vert0(v0.x, v0.y, v0.z);
    Eigen::Vector3f vert1(v1.x, v1.y, v1.z);
    Eigen::Vector3f vert2(v2.x, v2.y, v2.z);

    // Compute edges
    Eigen::Vector3f edge1 = vert1 - vert0;
    Eigen::Vector3f edge2 = vert2 - vert0;

    // Compute determinant
    Eigen::Vector3f h = direction.cross(edge2);
    float a = edge1.dot(h);

    if (a > -EPSILON && a < EPSILON) {
        return false; // Ray is parallel to triangle
    }

    float f = 1.0f / a;
    Eigen::Vector3f s = origin - vert0;
    float u = f * s.dot(h);

    if (u < 0.0f || u > 1.0f) {
        return false; // Intersection outside triangle
    }

    Eigen::Vector3f q = s.cross(edge1);
    float v = f * direction.dot(q);

    if (v < 0.0f || u + v > 1.0f) {
        return false; // Intersection outside triangle
    }

    // Compute t to find intersection point
    t = f * edge2.dot(q);
    if (t > EPSILON) { // Intersection in front of ray origin
        return true;
    }

    return false;
}

// Function to find the entry point and normal of the ray with the mesh
bool findRayMeshIntersection(const pcl::PolygonMesh& mesh, const pcl::PointXYZ& ray_origin,
                            const pcl::PointXYZ& ray_end, pcl::PointXYZ& intersection_point,
                            Eigen::Vector3f& normal, int skip_triangle_index = -1)
{
    // Convert mesh.cloud to PointCloud<PointXYZ>
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(mesh.cloud, *cloud);

    // Compute ray direction
    pcl::PointXYZ ray_direction(ray_end.x - ray_origin.x, ray_end.y - ray_origin.y, ray_end.z - ray_origin.z);
    Eigen::Vector3f ray_dir_eigen(ray_direction.x, ray_direction.y, ray_direction.z);
    ray_dir_eigen.normalize(); // Normalize for normal orientation

    float min_t = std::numeric_limits<float>::max();
    bool found = false;
    int triangle_index = 0;

    // Iterate through each triangle in the mesh
    for (const auto& polygon : mesh.polygons) {
        if (polygon.vertices.size() != 3 || triangle_index == skip_triangle_index) {
            triangle_index++;
            continue; // Skip non-triangular polygons or the specified triangle
        }

        // Get the three vertices of the triangle
        const pcl::PointXYZ& v0 = cloud->points[polygon.vertices[0]];
        const pcl::PointXYZ& v1 = cloud->points[polygon.vertices[1]];
        const pcl::PointXYZ& v2 = cloud->points[polygon.vertices[2]];

        float t;
        if (rayTriangleIntersection(ray_origin, ray_direction, v0, v1, v2, t)) {
            if (t < min_t) {
                min_t = t;
                // Compute intersection point: origin + t * direction
                intersection_point.x = ray_origin.x + t * ray_direction.x;
                intersection_point.y = ray_origin.y + t * ray_direction.y;
                intersection_point.z = ray_origin.z + t * ray_direction.z;

                // Compute triangle normal
                Eigen::Vector3f edge1(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
                Eigen::Vector3f edge2(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
                normal = edge1.cross(edge2).normalized();
                // Ensure normal points opposite to ray direction
                if (normal.dot(-ray_dir_eigen) < 0) {
                    normal = -normal;
                }

                found = true;
            }
        }
        triangle_index++;
    }

    return found;
}

// Function to compute refracted ray direction using Snell's Law
Eigen::Vector3f computeRefractedRay(const Eigen::Vector3f& incident_dir,
                                    const Eigen::Vector3f& normal,
                                    float n1, float n2)
{
    // Normalize incident direction
    Eigen::Vector3f I = incident_dir.normalized();
    Eigen::Vector3f N = normal.normalized();

    // Cosine of incident angle
    float cos_theta1 = -I.dot(N);
    if (cos_theta1 < 0) {
        // If ray is inside the medium, swap indices and flip normal
        std::swap(n1, n2);
        N = -N;
        cos_theta1 = -cos_theta1;
    }

    // Snell's Law: n1 * sin(theta1) = n2 * sin(theta2)
    float n = n1 / n2;
    float sin_theta1 = std::sqrt(std::max(0.0f, 1.0f - cos_theta1 * cos_theta1));
    float sin_theta2 = n * sin_theta1;

    // Check for total internal reflection
    if (sin_theta2 > 1.0f) {
        return Eigen::Vector3f(0, 0, 0); // No refraction, return zero vector
    }

    float cos_theta2 = std::sqrt(std::max(0.0f, 1.0f - sin_theta2 * sin_theta2));

    // Refracted direction: n * I + (n * cos_theta1 - cos_theta2) * N
    Eigen::Vector3f refracted_dir = n * I + (n * cos_theta1 - cos_theta2) * N;
    return refracted_dir.normalized();
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <mesh_file.ply>\n";
        return 1;
    }

    std::string mesh_file = argv[1];
    pcl::PolygonMesh mesh;

    // Load mesh using load_mesh function
    if (load_mesh(mesh_file, mesh) < 0) {
        return -1;
    }

    // Compute and print the mesh boundary
    auto [min_point, max_point] = getMeshBoundary(mesh);
    std::cout << "Mesh Boundary:\n";
    std::cout << "Min Point: (" << min_point.x << ", " << min_point.y << ", " << min_point.z << ")\n";
    std::cout << "Max Point: (" << max_point.x << ", " << max_point.y << ", " << max_point.z << ")\n";

    // Visualizer setup
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Mesh Viewer"));
    viewer->setBackgroundColor(0.1, 0.1, 0.1);
    viewer->addPolygonMesh(mesh, "mesh");

    // Define a 5x5 grid of rays centered around (0, -3, max_point.z/2)
    const int grid_size = 30; // 5x5 grid

    const float ray_length = (max_point.y - min_point.y)*1.4; // Length of incident ray (from y=-3 to y=3)
    const float second_ray_length = ray_length;
    
    // Center the grid around x=0    
    const float spacing_width = (max_point.x - min_point.x)/(grid_size - 1); // Spacing between rays
    const float spacing_height = (max_point.z-min_point.z)/(grid_size - 1); // Spacing between rays

    // Loop over the grid
    float y = min_point.y - 0.2 * min_point.y;
    for (int i = 0; i < grid_size; ++i) { // itterate over height
        for (int j = 0; j < grid_size; ++j) {

            // Compute ray start point
            float x = min_point.x + spacing_width * j;
            float z = min_point.z + spacing_height * i;



            pcl::PointXYZ start(x, y, z);
            pcl::PointXYZ end(x, y + ray_length, z);

            // Generate unique IDs for visualizer
            std::stringstream ss;
            ss << "ray_" << i << "_" << j;
            std::string ray_id = ss.str();
            ss.str(""); // Clear stringstream
            ss << "intersection_" << i << "_" << j;
            std::string intersection_id = ss.str();
            ss.str("");
            ss << "refracted_ray_" << i << "_" << j;
            std::string refracted_ray_id = ss.str();
            ss.str("");
            ss << "second_intersection_" << i << "_" << j;
            std::string second_intersection_id = ss.str();

            // ACHTUNG: First array
            // Add the ray (white)
            // viewer->addLine(start, end, 1.0, 1.0, 1.0, ray_id);

            // Find and visualize the first ray-mesh intersection
            pcl::PointXYZ intersection_point;
            Eigen::Vector3f normal;
            if (findRayMeshIntersection(mesh, start, end, intersection_point, normal)) {
                // std::cout << "Ray (" << i << ", " << j << ") First Intersection Point: ("
                //           << intersection_point.x << ", " << intersection_point.y << ", "
                //           << intersection_point.z << ")\n";
                // std::cout << "Surface Normal: (" << normal.x() << ", " << normal.y() << ", "
                //           << normal.z() << ")\n";


                // Add a red sphere at the first intersection point
                viewer->addSphere(intersection_point, 0.05, 1.0, 0.0, 0.0, intersection_id);

                // Compute refracted ray direction using Snell's Law
                Eigen::Vector3f incident_dir(end.x - start.x, end.y - start.y, end.z - start.z);
                float n1 = 1.0f; // Refractive index of air
                float n2 = 1.5f; // Refractive index of glass
                Eigen::Vector3f refracted_dir = computeRefractedRay(incident_dir, normal, n1, n2);

                if (refracted_dir.norm() > 0) {
                    // Visualize refracted ray (green, length scaled for visibility)
                    pcl::PointXYZ refracted_end;
                    float refracted_ray_vis_length = 3.0f; // Length for visualization
                    refracted_end.x = intersection_point.x + refracted_ray_vis_length * refracted_dir.x();
                    refracted_end.y = intersection_point.y + refracted_ray_vis_length * refracted_dir.y();
                    refracted_end.z = intersection_point.z + refracted_ray_vis_length * refracted_dir.z();
                    viewer->addLine(intersection_point, refracted_end, 0.0, 1.0, 0.0, refracted_ray_id);

                    // std::cout << "Refracted Ray Direction: (" << refracted_dir.x() << ", "
                    //           << refracted_dir.y() << ", " << refracted_dir.z() << ")\n";

                    // Calculate the next intersection with the refracted ray
                    pcl::PointXYZ second_intersection_point;
                    Eigen::Vector3f second_normal;
                    pcl::PointXYZ second_end;
                    second_end.x = intersection_point.x + second_ray_length * refracted_dir.x();
                    second_end.y = intersection_point.y + second_ray_length * refracted_dir.y();
                    second_end.z = intersection_point.z + second_ray_length * refracted_dir.z();

                    if (findRayMeshIntersection(mesh, intersection_point, second_end, second_intersection_point, second_normal)) {
                        // std::cout << "Ray (" << i << ", " << j << ") Second Intersection Point: ("
                        //           << second_intersection_point.x << ", " << second_intersection_point.y << ", "
                        //           << second_intersection_point.z << ")\n";

                        // Add a blue sphere at the second intersection point
                        viewer->addSphere(second_intersection_point, 0.05, 0.0, 0.0, 1.0, second_intersection_id);
                    } else {
                        // std::cout << "Ray (" << i << ", " << j << ") No second intersection found for the refracted ray.\n";
                        continue;
                    }
                } else {
                    // std::cout << "Ray (" << i << ", " << j << ") No refraction: Total internal reflection occurred.\n";
                    continue;
                }
            } else {
                // std::cout << "Ray (" << i << ", " << j << ") No first intersection found between the ray and the mesh.\n";
                continue;
            }
        }
    }

    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    // Rendering Loop
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }

    return 0;
}