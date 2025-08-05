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

#include "bodysystem.hpp"

#include <cuda_runtime.h>

#include <vector>

template <typename T> struct DeviceData {
    T*           dPos[2];    // mapped host pointers
    T*           dVel;
    cudaEvent_t  event;
    unsigned int offset;
    unsigned int numBodies;
};

// CUDA BodySystem: runs on the GPU
template <typename T> class BodySystemCUDA final : public BodySystem<T> {
 public:
    BodySystemCUDA(unsigned int numBodies, unsigned int numDevices, unsigned int blockSize, bool usePBO, bool useSysMem = false, bool useP2P = true, int deviceId = 0);
    virtual ~BodySystemCUDA();

    virtual void loadTipsyFile(const std::filesystem::path& filename) override;

    virtual void update(T deltaTime) override;

    virtual void setSoftening(T softening) override;
    virtual void setDamping(T damping) override;

    virtual T*   getArray(BodyArray array) override;
    virtual void setArray(BodyArray array, std::span<const T> data) override;

    virtual unsigned int getCurrentReadBuffer() const override { return m_pbo[m_currentRead]; }

    virtual unsigned int getNumBodies() const override { return m_numBodies; }

 private:    // methods
    BodySystemCUDA() = default;

    virtual void _initialize(int numBodies) override;
    virtual void _finalize() noexcept override;

    unsigned int m_numBodies;
    unsigned int m_numDevices;
    bool         m_bInitialized;
    int          m_devID;

    // Host data
    T* m_hPos[2];
    T* m_hVel;

    std::vector<DeviceData<T>> m_deviceData;

    bool         m_bUsePBO;
    bool         m_bUseSysMem;
    bool         m_bUseP2P;
    unsigned int m_SMVersion;

    T m_damping;

    unsigned int          m_pbo[2];
    cudaGraphicsResource* m_pGRes[2];
    unsigned int          m_currentRead;
    unsigned int          m_currentWrite;

    unsigned int m_blockSize;
};

extern template BodySystemCUDA<float>;
extern template BodySystemCUDA<double>;