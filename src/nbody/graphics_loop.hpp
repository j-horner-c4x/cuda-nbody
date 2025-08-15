#pragma once

struct ComputeConfig;
struct InterfaceConfig;
class Camera;
class Controls;
class ParticleRenderer;

auto execute_graphics_loop(ComputeConfig& compute, InterfaceConfig& interface, Camera& camera, Controls& controls, ParticleRenderer& renderer) -> void;