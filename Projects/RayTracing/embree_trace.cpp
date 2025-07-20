#include <cstring>

#include <embree3/rtcore.h>
#include <iostream>
#include <vector>
#include <cmath>

// Mesh structure
struct Triangle {
    float v0[3], v1[3], v2[3];
};

// Normalize vector
void normalize(float* v) {
    float len = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    for (int i = 0; i < 3; ++i) v[i] /= len;
}

// Refract ray using Snell's law
bool refract(const float* I, const float* N, float n1, float n2, float* T) {
    float cosi = -std::max(-1.0f, std::min(1.0f, I[0]*N[0] + I[1]*N[1] + I[2]*N[2]));
    float eta = n1 / n2;
    float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
    if (k < 0.0f) return false; // Total internal reflection
    for (int i = 0; i < 3; ++i)
        T[i] = eta * I[i] + (eta * cosi - std::sqrt(k)) * N[i];
    normalize(T);
    return true;
}

int main() {
    // Setup Embree device and scene
    RTCDevice device = rtcNewDevice(nullptr);
    RTCScene scene = rtcNewScene(device);

    // Define a test triangle (replace with mesh loader later)
    Triangle tri = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}
    };

    // Create geometry
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
    float* vertices = (float*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
                                                      RTC_FORMAT_FLOAT3, 3 * sizeof(float), 3);
    unsigned* indices = (unsigned*)rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0,
                                                           RTC_FORMAT_UINT3, 3 * sizeof(unsigned), 1);

    memcpy(&vertices[0], tri.v0, 3 * sizeof(float));
    memcpy(&vertices[3], tri.v1, 3 * sizeof(float));
    memcpy(&vertices[6], tri.v2, 3 * sizeof(float));
    indices[0] = 0; indices[1] = 1; indices[2] = 2;

    rtcCommitGeometry(geom);
    unsigned geomID = rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);

    // Define ray from light at (3,1,0) toward origin (just an example)
    RTCRayHit rayhit;
    rayhit.ray.org_x = 3.0f; rayhit.ray.org_y = 1.0f; rayhit.ray.org_z = 0.0f;
    rayhit.ray.dir_x = -3.0f; rayhit.ray.dir_y = -1.0f; rayhit.ray.dir_z = -1.0f;
    normalize(&rayhit.ray.dir_x);

    rayhit.ray.tnear = 0.0f;
    rayhit.ray.tfar = 1000.0f;
    rayhit.ray.mask = -1;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

    // Intersect ray
    RTCIntersectContext context;
    rtcInitIntersectContext(&context);
    rtcIntersect1(scene, &context, &rayhit);

    if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
        std::cout << "Ray hit triangle at distance: " << rayhit.ray.tfar << std::endl;

        // Calculate surface normal from barycentrics
        float u = rayhit.hit.u, v = rayhit.hit.v;
        float n[3] = {0, 0, 1}; // Triangle is flat on z=0

        // Compute refraction direction
        float out_dir[3];
        if (refract(&rayhit.ray.dir_x, n, 1.0f, 1.5f, out_dir)) {
            std::cout << "Refracted direction: (" << out_dir[0] << ", " << out_dir[1] << ", " << out_dir[2] << ")\n";
        } else {
            std::cout << "Total internal reflection occurred.\n";
        }

    } else {
        std::cout << "Ray missed geometry.\n";
    }

    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
    return 0;
}
