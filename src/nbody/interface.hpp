#pragma once

#include "paramgl.hpp"
#include "render_particles.hpp"

#include <memory>

class ParamListGL;
class Camera;
struct ComputeConfig;
class ParticleRenderer;

struct InterfaceConfig {
    bool                          display_enabled;
    bool                          show_sliders;
    std::unique_ptr<ParamListGL>  param_list;
    bool                          full_screen;
    bool                          display_interactions;
    ParticleRenderer::DisplayMode display_mode;
    int                           fps_count;
    int                           fps_limit;

    auto toggle_sliders() noexcept -> void { show_sliders = !show_sliders; }
    auto toggle_interactions() noexcept -> void { display_interactions = !display_interactions; }
    auto cycle_display_mode() noexcept -> void { display_mode = (ParticleRenderer::DisplayMode)((display_mode + 1) % ParticleRenderer::PARTICLE_NUM_MODES); }
    auto togle_display() noexcept -> void { display_enabled = !display_enabled; }

    auto display(ComputeConfig& compute, Camera& camera, ParticleRenderer& renderer) -> void;
};