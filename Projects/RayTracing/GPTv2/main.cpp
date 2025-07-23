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
#include <vector>
#include <fstream>

// Function to load mesh from a .ply file
int load_mesh(const std::string& mesh_file, pcl::PolygonMesh& mesh)
{
    if (pcl::io::loadPLYFile(mesh_file, mesh) < 0) {
        std::cerr << "Error loading mesh file: " << mesh_file << "\n";
        return -1;
    }
    return 0;
}

// Function to compute the axis-aligned bounding box (AABB) of the mesh
std::pair<pcl::PointXYZ, pcl::PointXYZ> getMeshBoundary(const pcl::PolygonMesh& mesh)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(mesh.cloud, *cloud);

    if (cloud->points.empty()) {
        std::cerr << "Error: Mesh has no vertices.\n";
        return std::make_pair(pcl::PointXYZ(0, 0, 0), pcl::PointXYZ(0, 0, 0));
    }

    pcl::PointXYZ min_point = cloud->points[0];
    pcl::PointXYZ max_point = cloud->points[0];

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

    Eigen::Vector3f origin(ray_origin.x, ray_origin.y, ray_origin.z);
    Eigen::Vector3f direction(ray_direction.x, ray_direction.y, ray_direction.z);
    Eigen::Vector3f vert0(v0.x, v0.y, v0.z);
    Eigen::Vector3f vert1(v1.x, v1.y, v1.z);
    Eigen::Vector3f vert2(v2.x, v2.y, v2.z);

    Eigen::Vector3f edge1 = vert1 - vert0;
    Eigen::Vector3f edge2 = vert2 - vert0;

    Eigen::Vector3f h = direction.cross(edge2);
    float a = edge1.dot(h);

    if (a > -EPSILON && a < EPSILON) {
        return false;
    }

    float f = 1.0f / a;
    Eigen::Vector3f s = origin - vert0;
    float u = f * s.dot(h);

    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    Eigen::Vector3f q = s.cross(edge1);
    float v = f * direction.dot(q);

    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    t = f * edge2.dot(q);
    if (t > EPSILON) {
        return true;
    }

    return false;
}

// Function to find the entry point and normal of the ray with the mesh
bool findRayMeshIntersection(const pcl::PolygonMesh& mesh, const pcl::PointXYZ& ray_origin,
                            const pcl::PointXYZ& ray_end, pcl::PointXYZ& intersection_point,
                            Eigen::Vector3f& normal, int skip_triangle_index = -1)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(mesh.cloud, *cloud);

    pcl::PointXYZ ray_direction(ray_end.x - ray_origin.x, ray_end.y - ray_origin.y, ray_end.z - ray_origin.z);
    Eigen::Vector3f ray_dir_eigen(ray_direction.x, ray_direction.y, ray_direction.z);
    ray_dir_eigen.normalize();

    float min_t = std::numeric_limits<float>::max();
    bool found = false;
    int triangle_index = 0;

    for (const auto& polygon : mesh.polygons) {
        if (polygon.vertices.size() != 3 || triangle_index == skip_triangle_index) {
            triangle_index++;
            continue;
        }

        const pcl::PointXYZ& v0 = cloud->points[polygon.vertices[0]];
        const pcl::PointXYZ& v1 = cloud->points[polygon.vertices[1]];
        const pcl::PointXYZ& v2 = cloud->points[polygon.vertices[2]];

        float t;
        if (rayTriangleIntersection(ray_origin, ray_direction, v0, v1, v2, t)) {
            if (t < min_t) {
                min_t = t;
                intersection_point.x = ray_origin.x + t * ray_direction.x;
                intersection_point.y = ray_origin.y + t * ray_direction.y;
                intersection_point.z = ray_origin.z + t * ray_direction.z;

                Eigen::Vector3f edge1(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
                Eigen::Vector3f edge2(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
                normal = edge1.cross(edge2).normalized();
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
    Eigen::Vector3f I = incident_dir.normalized();
    Eigen::Vector3f N = normal.normalized();

    float cos_theta1 = -I.dot(N);
    if (cos_theta1 < 0) {
        std::swap(n1, n2);
        N = -N;
        cos_theta1 = -cos_theta1;
    }

    float n = n1 / n2;
    float sin_theta1 = std::sqrt(std::max(0.0f, 1.0f - cos_theta1 * cos_theta1));
    float sin_theta2 = n * sin_theta1;

    if (sin_theta2 > 1.0f) {
        return Eigen::Vector3f(0, 0, 0);
    }

    float cos_theta2 = std::sqrt(std::max(0.0f, 1.0f - sin_theta2 * sin_theta2));
    Eigen::Vector3f refracted_dir = n * I + (n * cos_theta1 - cos_theta2) * N;
    return refracted_dir.normalized();
}

// Function to write a PPM image
void writePPM(const std::vector<std::vector<bool>>& image, int width, int height, const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }

    file << "P3\n" << width << " " << height << "\n255\n";
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (image[y][x]) {
                file << "255 255 255 ";
            } else {
                file << "0 0 0 ";
            }
        }
        file << "\n";
    }
    file.close();
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <mesh_file.ply>\n";
        return 1;
    }

    std::string mesh_file = argv[1];
    pcl::PolygonMesh mesh;

    if (load_mesh(mesh_file, mesh) < 0) {
        return -1;
    }

    auto [min_point, max_point] = getMeshBoundary(mesh);
    std::cout << "Mesh Boundary:\n";
    std::cout << "Min Point: (" << min_point.x << ", " << min_point.y << ", " << min_point.z << ")\n";
    std::cout << "Max Point: (" << max_point.x << ", " << max_point.y << ", " << max_point.z << ")\n";

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Mesh Viewer"));
    viewer->setBackgroundColor(0.1, 0.1, 0.1);
    viewer->addPolygonMesh(mesh, "mesh");

    // Camera plane at y = 1.1f
    float camera_y = 1.1f;

    // Define image resolution
    const int image_width = 512;
    const int image_height = 512;

    // Compute camera plane extent based on mesh boundary
    float plane_min_x = min_point.x;
    float plane_max_x = max_point.x;
    float plane_min_z = min_point.z;
    float plane_max_z = max_point.z;

    float plane_width = plane_max_x - plane_min_x;
    float plane_height = plane_max_z - plane_min_z;

    // Ray direction (orthographic projection, along -y)
    Eigen::Vector3f ray_dir(0.0f, -1.0f, 0.0f);

    // Initialize image
    std::vector<std::vector<bool>> image(image_height, std::vector<bool>(image_width, false));

    // Generate rays and check intersections
    for (int y = 0; y < image_height; ++y) {
        for (int x = 0; x < image_width; ++x) {
            float u = static_cast<float>(x) / (image_width - 1);
            float v = static_cast<float>(y) / (image_height - 1);
            float ray_x = plane_min_x + u * plane_width;
            float ray_z = plane_min_z + v * plane_height;
            pcl::PointXYZ ray_origin(ray_x, camera_y, ray_z);

            pcl::PointXYZ ray_end = ray_origin;
            ray_end.y -= 10.0f; // Extend ray downward

            pcl::PointXYZ intersection_point;
            Eigen::Vector3f normal;
            if (findRayMeshIntersection(mesh, ray_origin, ray_end, intersection_point, normal)) {
                image[y][x] = true; // Mark pixel as hit
            }
        }
    }

    // Save the rendered image
    writePPM(image, image_width, image_height, "rendered_image.ppm");
    std::cout << "Rendered image saved as 'rendered_image.ppm'\n";

    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }

    return 0;
}