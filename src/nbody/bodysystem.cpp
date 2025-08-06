#include "bodysystem.hpp"

#include <vector_types.h>

namespace {
auto normalize(float3& vector) -> float {
    const auto dist = sqrtf(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);

    if (dist > 1e-6) {
        vector.x /= dist;
        vector.y /= dist;
        vector.z /= dist;
    }

    return dist;
}

constexpr auto dot(float3 v0, float3 v1) noexcept -> float {
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

constexpr auto cross(float3 v0, float3 v1) noexcept -> float3 {
    float3 rt;
    rt.x = v0.y * v1.z - v0.z * v1.y;
    rt.y = v0.z * v1.x - v0.x * v1.z;
    rt.z = v0.x * v1.y - v0.y * v1.x;
    return rt;
}
}    // namespace

template <std::floating_point T> void randomizeBodies(NBodyConfig config, T* pos, T* vel, float* color, float clusterScale, float velocityScale, int numBodies, bool vec4vel) {
    using enum NBodyConfig;

    switch (config) {
        default:
        case NBODY_CONFIG_RANDOM:
            {
                float scale  = clusterScale * std::max<float>(1.0f, numBodies / (1024.0f));
                float vscale = velocityScale * scale;

                int p = 0, v = 0;
                int i = 0;

                while (i < numBodies) {
                    float3 point;
                    // const int scale = 16;
                    point.x      = rand() / (float)RAND_MAX * 2 - 1;
                    point.y      = rand() / (float)RAND_MAX * 2 - 1;
                    point.z      = rand() / (float)RAND_MAX * 2 - 1;
                    float lenSqr = dot(point, point);

                    if (lenSqr > 1)
                        continue;

                    float3 velocity;
                    velocity.x = rand() / (float)RAND_MAX * 2 - 1;
                    velocity.y = rand() / (float)RAND_MAX * 2 - 1;
                    velocity.z = rand() / (float)RAND_MAX * 2 - 1;
                    lenSqr     = dot(velocity, velocity);

                    if (lenSqr > 1)
                        continue;

                    pos[p++] = point.x * scale;    // pos.x
                    pos[p++] = point.y * scale;    // pos.y
                    pos[p++] = point.z * scale;    // pos.z
                    pos[p++] = 1.0f;               // mass

                    vel[v++] = velocity.x * vscale;    // pos.x
                    vel[v++] = velocity.y * vscale;    // pos.x
                    vel[v++] = velocity.z * vscale;    // pos.x

                    if (vec4vel)
                        vel[v++] = 1.0f;    // inverse mass

                    i++;
                }
            }
            break;

        case NBODY_CONFIG_SHELL:
            {
                float scale  = clusterScale;
                float vscale = scale * velocityScale;
                float inner  = 2.5f * scale;
                float outer  = 4.0f * scale;

                int p = 0, v = 0;
                int i = 0;

                while (i < numBodies)    // for(int i=0; i < numBodies; i++)
                {
                    float x, y, z;
                    x = rand() / (float)RAND_MAX * 2 - 1;
                    y = rand() / (float)RAND_MAX * 2 - 1;
                    z = rand() / (float)RAND_MAX * 2 - 1;

                    float3 point = {x, y, z};
                    float  len   = normalize(point);

                    if (len > 1)
                        continue;

                    pos[p++] = point.x * (inner + (outer - inner) * rand() / (float)RAND_MAX);
                    pos[p++] = point.y * (inner + (outer - inner) * rand() / (float)RAND_MAX);
                    pos[p++] = point.z * (inner + (outer - inner) * rand() / (float)RAND_MAX);
                    pos[p++] = 1.0f;

                    x           = 0.0f;    // * (rand() / (float) RAND_MAX * 2 - 1);
                    y           = 0.0f;    // * (rand() / (float) RAND_MAX * 2 - 1);
                    z           = 1.0f;    // * (rand() / (float) RAND_MAX * 2 - 1);
                    float3 axis = {x, y, z};
                    normalize(axis);

                    if (1 - dot(point, axis) < 1e-6) {
                        axis.x = point.y;
                        axis.y = point.x;
                        normalize(axis);
                    }

                    // if (point.y < 0) axis = scalevec(axis, -1);
                    float3 vv = {(float)pos[4 * i], (float)pos[4 * i + 1], (float)pos[4 * i + 2]};
                    vv        = cross(vv, axis);
                    vel[v++]  = vv.x * vscale;
                    vel[v++]  = vv.y * vscale;
                    vel[v++]  = vv.z * vscale;

                    if (vec4vel)
                        vel[v++] = 1.0f;

                    i++;
                }
            }
            break;

        case NBODY_CONFIG_EXPAND:
            {
                float scale = clusterScale * numBodies / (1024.f);

                if (scale < 1.0f)
                    scale = clusterScale;

                float vscale = scale * velocityScale;

                int p = 0, v = 0;

                for (int i = 0; i < numBodies;) {
                    float3 point;

                    point.x = rand() / (float)RAND_MAX * 2 - 1;
                    point.y = rand() / (float)RAND_MAX * 2 - 1;
                    point.z = rand() / (float)RAND_MAX * 2 - 1;

                    float lenSqr = dot(point, point);

                    if (lenSqr > 1)
                        continue;

                    pos[p++] = point.x * scale;     // pos.x
                    pos[p++] = point.y * scale;     // pos.y
                    pos[p++] = point.z * scale;     // pos.z
                    pos[p++] = 1.0f;                // mass
                    vel[v++] = point.x * vscale;    // pos.x
                    vel[v++] = point.y * vscale;    // pos.x
                    vel[v++] = point.z * vscale;    // pos.x

                    if (vec4vel)
                        vel[v++] = 1.0f;    // inverse mass

                    i++;
                }
            }
            break;
    }

    if (color) {
        int v = 0;

        for (int i = 0; i < numBodies; i++) {
            // const int scale = 16;
            color[v++] = rand() / (float)RAND_MAX;
            color[v++] = rand() / (float)RAND_MAX;
            color[v++] = rand() / (float)RAND_MAX;
            color[v++] = 1.0f;
        }
    }
}

template void randomizeBodies<float>(NBodyConfig config, float* pos, float* vel, float* color, float clusterScale, float velocityScale, int numBodies, bool vec4vel);
template void randomizeBodies<double>(NBodyConfig config, double* pos, double* vel, float* color, float clusterScale, float velocityScale, int numBodies, bool vec4vel);