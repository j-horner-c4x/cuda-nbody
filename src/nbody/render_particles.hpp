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
    ParticleRenderer(std::size_t nb_bodies, bool fp64);

    auto colour() noexcept -> std::span<float> { return colour_; }

    // invoked by CPU impl or GPU impl using host memory
    void set_positions(std::span<const float> pos);
    void set_positions(std::span<const double> pos);
    // invoked by GPU impl
    void set_pbo(unsigned int pbo, bool fp64);

    enum DisplayMode { PARTICLE_POINTS, PARTICLE_SPRITES, PARTICLE_SPRITES_COLOR, PARTICLE_NUM_MODES };

    void display(DisplayMode mode, float sprite_size);

 private:    // methods
    void resetPBO();

    void _initGL();
    void _createTexture();
    void _drawPoints(bool color);

    std::vector<float> colour_;

    std::span<const float>  pos_;
    std::span<const double> pos_fp64_;

    unsigned int program_points_  = 0;
    unsigned int program_sprites_ = 0;
    unsigned int texture_         = 0;
    unsigned int pbo_             = 0;
    unsigned int vbo_colour_      = 0;

    bool fp64_positions_ = false;
};
