#pragma once

struct ComputeConfig;
struct InterfaceConfig;
struct CameraConfig;
struct ControlsConfig;

auto execute_graphics_loop(ComputeConfig& compute, InterfaceConfig& interface, CameraConfig& camera, ControlsConfig& controls) -> void;