#pragma once

#include <array>
#include <memory>

class ParamListGL;

////////////////////////////////////////
// Demo Parameters
////////////////////////////////////////
struct NBodyParams {
    float                m_timestep;
    float                m_clusterScale;
    float                m_velocityScale;
    float                m_softening;
    float                m_damping;
    float                m_pointSize;
    std::array<float, 3> camera_origin;

    auto print() const -> void;

    auto create_sliders() -> std::unique_ptr<ParamListGL>;
};