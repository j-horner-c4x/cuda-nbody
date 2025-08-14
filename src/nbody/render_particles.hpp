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

#include <array>
#include <span>
#include <vector>

class ParticleRenderer {
 public:
    ParticleRenderer(std::size_t nb_bodies, float point_size, bool fp64);

    auto colour() noexcept -> std::span<float> { return colour_; }

    auto reset(bool fp64, float point_size) -> void;

    // invoked by CPU impl
    void setPositions(std::span<float> pos);
    void setPositions(std::span<double> pos);
    // invoked by GPU impl
    void setPBO(unsigned int pbo, int numParticles, bool fp64);

    auto reset(std::span<const float> colour, bool fp64, float point_size) -> void;

    void setBaseColor(const std::array<float, 4>& colour) { m_baseColor = colour; }
    void setColours(std::span<const float> colour);

    enum DisplayMode { PARTICLE_POINTS, PARTICLE_SPRITES, PARTICLE_SPRITES_COLOR, PARTICLE_NUM_MODES };

    void display(DisplayMode mode = PARTICLE_POINTS);

    void setPointSize(float size) { m_pointSize = size; }
    void setSpriteSize(float size) { m_spriteSize = size; }

 private:    // methods
    void resetPBO();

    void _initGL();
    void _createTexture();
    void _drawPoints(bool color);

    std::vector<float> colour_;

    float*  m_pos = nullptr;
    double* m_pos_fp64;
    int     m_numParticles = 0;

    float m_pointSize  = 1.f;
    float m_spriteSize = 2.f;

    unsigned int m_programPoints  = 0;
    unsigned int m_programSprites = 0;
    unsigned int m_texture        = 0;
    unsigned int m_pbo            = 0;
    unsigned int m_vboColor       = 0;

    std::array<float, 4> m_baseColor;

    bool m_bFp64Positions = false;
};
