#pragma once

struct ComputeConfig;
struct InterfaceConfig;
class Camera;
struct ControlsConfig;
class ParticleRenderer;

auto execute_graphics_loop(ComputeConfig& compute, InterfaceConfig& interface, Camera& camera, ControlsConfig& controls, ParticleRenderer& renderer) -> void;