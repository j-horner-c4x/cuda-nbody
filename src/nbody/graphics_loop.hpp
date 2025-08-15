#pragma once

struct ComputeConfig;
class Interface;
class Camera;
class Controls;
class ParticleRenderer;

auto execute_graphics_loop(ComputeConfig& compute, Interface& interface, Camera& camera, Controls& controls, ParticleRenderer& renderer) -> void;