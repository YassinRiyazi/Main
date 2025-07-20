#include <embree3/rtcore.h>
#include <pcl/io/ply_io.h>
#include <pcl/PolygonMesh.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common.h>
#include <iostream>
#include <vector>
#include <limits>

void errorHandler(void* userPtr, RTCError code, const char* str) {
    std::cerr << "Embree Error [" << code << "]: " << (str ? str : "Unknown") << std::endl;
    exit(1);
}

// Function to compute reflected ray direction
void computeReflection(const float* incident, const float* normal, float* reflected) {
    float dot = incident[0] * normal[0] + incident[1] * normal[1] + incident[2] * normal[2];
    for (int i = 0; i < 3; ++i) {
        reflected[i] = incident[i] - 2.0f * dot * normal[i];
    }
}

int main(int argc, char** argv) {
    // Initialize Embree
    RTCDevice device = rtcNewDevice(nullptr);
    rtcSetDeviceErrorFunction(device, errorHandler, nullptr);
    RTCScene scene = rtcNewScene(device);
    rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);
    rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_HIGH);

    // Load PLY mesh using PCL
    pcl::PolygonMesh mesh;
    std::string ply_file = "/home/ysn-u25/Desktop/Main/Projects/RayTracing/Poly/hemisphere_mesh.ply";
    if (pcl::io::loadPLYFile(ply_file, mesh) < 0) {
        std::cerr << "Failed to load PLY file: " << ply_file << std::endl;
        rtcReleaseScene(scene);
        rtcReleaseDevice(device);
        return -1;
    }

    // Extract vertices and faces from PCL mesh
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromPCLPointCloud2(mesh.cloud, cloud);
    std::vector<unsigned int> indices;
    for (const auto& polygon : mesh.polygons) {
        for (const auto& idx : polygon.vertices) {
            indices.push_back(idx);
        }
    }

    // Debug: Print mesh info
    std::cout << "Loaded mesh with " << cloud.size() << " vertices and " << indices.size() / 3 << " triangles" << std::endl;

    // Compute and print bounding box
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(cloud, min_pt, max_pt);
    std::cout << "Mesh bounding box: "
              << "Min (" << min_pt.x << ", " << min_pt.y << ", " << min_pt.z << "), "
              << "Max (" << max_pt.x << ", " << max_pt.y << ", " << max_pt.z << ")" << std::endl;

    // Create Embree geometry
    RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
    float* vertices = (float*)rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), cloud.size());
    for (size_t i = 0; i < cloud.size(); ++i) {
        vertices[i * 3 + 0] = cloud[i].x;
        vertices[i * 3 + 1] = cloud[i].y;
        vertices[i * 3 + 2] = cloud[i].z;
    }
    unsigned* triangles = (unsigned*)rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned), indices.size() / 3);
    for (size_t i = 0; i < indices.size(); ++i) {
        triangles[i] = indices[i];
    }

    rtcCommitGeometry(geometry);
    rtcAttachGeometry(scene, geometry);
    rtcReleaseGeometry(geometry);
    rtcCommitScene(scene);

    // Set up ray from light source (-1, -2, 0) to point (2, 1, 0)
    RTCRayHit rayhit;
    float light_pos[3] = {-1.0f, -2.0f, 0.0f};
    float hit_pos[3] = {2.0f, 1.0f, 0.0f};
    float incident_dir[3] = {hit_pos[0] - light_pos[0], hit_pos[1] - light_pos[1], hit_pos[2] - light_pos[2]};
    float length = sqrt(incident_dir[0] * incident_dir[0] + incident_dir[1] * incident_dir[1] + incident_dir[2] * incident_dir[2]);
    for (int i = 0; i < 3; ++i) {
        incident_dir[i] /= length; // Normalize direction
    }

    rayhit.ray.org_x = light_pos[0];
    rayhit.ray.org_y = light_pos[1];
    rayhit.ray.org_z = light_pos[2];
    rayhit.ray.dir_x = incident_dir[0];
    rayhit.ray.dir_y = incident_dir[1];
    rayhit.ray.dir_z = incident_dir[2];
    rayhit.ray.tnear = 0.0f;
    rayhit.ray.tfar = length; // Limit to distance to (2, 1, 0)
    rayhit.ray.mask = 0xFFFFFFFF;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

    // Intersect ray with scene
    RTCIntersectContext context;
    rtcInitIntersectContext(&context);
    rtcIntersect1(scene, &context, &rayhit);

    // Prepare visualization
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Hemisphere Mesh Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // Add transparent mesh
    viewer->addPolygonMesh(mesh, "mesh");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "mesh");

    // Add light source as a yellow sphere
    pcl::PointXYZ light_point(light_pos[0], light_pos[1], light_pos[2]);
    // viewer->addSphere(light_point, 0.1, 1.0, 1.0, 0.0, "light_source");

    // Add incident ray as a green line
    pcl::PointXYZ ray_start(light_pos[0], light_pos[1], light_pos[2]);
    pcl::PointXYZ ray_end(hit_pos[0], hit_pos[1], hit_pos[2]);
    viewer->addLine(ray_start, ray_end, 0.0, 1.0, 0.0, "incident_ray");
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "incident_ray");

    // Handle ray intersection
    if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
        // Actual intersection point
        pcl::PointXYZ hit_point;
        hit_point.x = rayhit.ray.org_x + rayhit.ray.tfar * rayhit.ray.dir_x;
        hit_point.y = rayhit.ray.org_y + rayhit.ray.tfar * rayhit.ray.dir_y;
        hit_point.z = rayhit.ray.org_z + rayhit.ray.tfar * rayhit.ray.dir_z;
        std::cout << "Ray hit at: (" << hit_point.x << ", " << hit_point.y << ", " << hit_point.z << ")" << std::endl;

        // Add intersection point to visualization
        pcl::PointCloud<pcl::PointXYZ>::Ptr hit_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        hit_cloud->push_back(hit_point);
        viewer->addPointCloud(hit_cloud, "hit_point");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "hit_point");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "hit_point");

        // Compute and visualize reflected ray
        float normal[3] = {rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z};
        float normal_length = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
        for (int i = 0; i < 3; ++i) {
            normal[i] /= normal_length; // Normalize normal
        }
        float reflected_dir[3];
        computeReflection(incident_dir, normal, reflected_dir);
        std::cout << "Reflected direction: (" << reflected_dir[0] << ", " << reflected_dir[1] << ", " << reflected_dir[2] << ")" << std::endl;

        // Visualize reflected ray (extend 2 units from hit point)
        pcl::PointXYZ reflect_end(hit_point.x + 2.0f * reflected_dir[0],
                                  hit_point.y + 2.0f * reflected_dir[1],
                                  hit_point.z + 2.0f * reflected_dir[2]);
        viewer->addLine(hit_point, reflect_end, 0.0, 0.0, 1.0, "reflected_ray");
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "reflected_ray");
    } else {
        std::cout << "Ray missed geometry. Point (2, 1, 0) may not be on the mesh." << std::endl;
    }

    // Adjust camera to focus on the mesh
    float center_x = (min_pt.x + max_pt.x) / 2.0f;
    float center_y = (min_pt.y + max_pt.y) / 2.0f;
    float center_z = (min_pt.z + max_pt.z) / 2.0f;
    float size = std::max({max_pt.x - min_pt.x, max_pt.y - min_pt.y, max_pt.z - min_pt.z, 4.0f});
    viewer->setCameraPosition(center_x, center_y, center_z + size * 2, // Camera position
                             center_x, center_y, center_z,           // Focal point
                             0, 1, 0);                               // Up vector
    viewer->addCoordinateSystem(size / 2.0); // Scale axes to mesh size

    // Run viewer
    viewer->initCameraParameters();
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }

    // Cleanup
    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
    return 0;
}