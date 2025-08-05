/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <vector_types.h>

#include <algorithm>
#include <filesystem>

enum class NBodyConfig { NBODY_CONFIG_RANDOM, NBODY_CONFIG_SHELL, NBODY_CONFIG_EXPAND, NBODY_NUM_CONFIGS };

enum class BodyArray {
    BODYSYSTEM_POSITION,
    BODYSYSTEM_VELOCITY,
};

template <std::floating_point T> using vec3 = std::conditional_t<std::is_same_v<T, float>, float3, std::conditional_t<std::is_same_v<T, double>, double3, std::array<T, 3>>>;
template <std::floating_point T> using vec4 = std::conditional_t<std::is_same_v<T, float>, float4, std::conditional_t<std::is_same_v<T, double>, double4, std::array<T, 4>>>;

// BodySystem abstract base class
template <std::floating_point T> class BodySystem {
 public:    // methods
    virtual void loadTipsyFile(const std::filesystem::path& filename) = 0;

    virtual void update(T deltaTime) = 0;

    virtual void setSoftening(T softening) = 0;
    virtual void setDamping(T damping)     = 0;

    virtual T*   getArray(BodyArray array)                = 0;
    virtual void setArray(BodyArray array, const T* data) = 0;

    virtual unsigned int getCurrentReadBuffer() const = 0;

    virtual unsigned int getNumBodies() const = 0;

    virtual void synchronizeThreads() const {};

    virtual ~BodySystem() = default;

 protected:                    // methods
    BodySystem() = default;    // default constructor

    virtual void _initialize(int numBodies) = 0;
    virtual void _finalize()                = 0;
};

// utility function
template <std::floating_point T> void randomizeBodies(NBodyConfig config, T* pos, T* vel, float* color, float clusterScale, float velocityScale, int numBodies, bool vec4vel);

extern template void randomizeBodies<float>(NBodyConfig config, float* pos, float* vel, float* color, float clusterScale, float velocityScale, int numBodies, bool vec4vel);
extern template void randomizeBodies<double>(NBodyConfig config, double* pos, double* vel, float* color, float clusterScale, float velocityScale, int numBodies, bool vec4vel);