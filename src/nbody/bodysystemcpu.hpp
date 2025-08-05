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

#include <span>
#include <vector>

// CPU Body System
template <std::floating_point T> class BodySystemCPU final : public BodySystem<T> {
 public:
    BodySystemCPU(int numBodies);
    virtual ~BodySystemCPU() = default;

    virtual void loadTipsyFile(const std::filesystem::path& filename) override;

    virtual void update(T deltaTime) override;

    virtual void setSoftening(T softening) override { m_softeningSquared = softening * softening; }
    virtual void setDamping(T damping) override { m_damping = damping; }

    virtual T*   getArray(BodyArray array) override;
    virtual void setArray(BodyArray array, std::span<const T> data) override;

    virtual unsigned int getCurrentReadBuffer() const override { return 0; }

    virtual unsigned int getNumBodies() const override { return m_numBodies; }

 private:                 // methods
    BodySystemCPU() {}    // default constructor

    virtual void _initialize(int numBodies) override;
    virtual void _finalize() noexcept override {};

    void _computeNBodyGravitation();
    void _integrateNBodySystem(T deltaTime);

    int m_numBodies;

    std::vector<T> m_pos;
    std::vector<T> m_vel;
    std::vector<T> m_force;

    T m_softeningSquared;
    T m_damping;
};

extern template BodySystemCPU<float>;
extern template BodySystemCPU<double>;