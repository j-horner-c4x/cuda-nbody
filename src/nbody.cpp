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

#include "git_commit_id.hpp"
#include "nbody/helper_gl.hpp"
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define NOMINMAX
#include <GL/wglew.h>
#endif

#include "nbody/bodysystemcpu.hpp"
#include "nbody/bodysystemcuda.hpp"
#include "nbody/camera.hpp"
#include "nbody/compute.hpp"
#include "nbody/controls.hpp"
#include "nbody/helper_cuda.hpp"
#include "nbody/interface.hpp"
#include "nbody/nbody_demo.hpp"
#include "nbody/paramgl.hpp"
#include "nbody/params.hpp"
#include "nbody/render_particles.hpp"
#include "nbody/win_coords.hpp"

#include <CLI/CLI.hpp>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <filesystem>
#include <format>
#include <memory>
#include <print>
#include <string_view>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

using Clock        = std::chrono::steady_clock;
using TimePoint    = std::chrono::time_point<Clock>;
using MilliSeconds = std::chrono::duration<float, std::milli>;

using std::ranges::copy;

template <std::floating_point T> auto compare_results(int num_bodies, BodySystemCUDA<T>& nbodyCuda) -> bool {
    bool passed = true;

    nbodyCuda.update(0.001f);

    {
        using enum BodyArray;

        auto nbodyCpu = BodySystemCPU<T>(num_bodies);

        nbodyCpu.setArray(BODYSYSTEM_POSITION, std::span{nbodyCuda.getArray(BODYSYSTEM_POSITION), static_cast<std::size_t>(num_bodies) * 4});
        nbodyCpu.setArray(BODYSYSTEM_VELOCITY, std::span{nbodyCuda.getArray(BODYSYSTEM_VELOCITY), static_cast<std::size_t>(num_bodies) * 4});

        nbodyCpu.update(0.001f);

        T* cudaPos = nbodyCuda.getArray(BODYSYSTEM_POSITION);
        T* cpuPos  = nbodyCpu.getArray(BODYSYSTEM_POSITION);

        T tolerance = 0.0005f;

        for (int i = 0; i < num_bodies; i++) {
            if (std::abs(cpuPos[i] - cudaPos[i]) > tolerance) {
                passed = false;
                std::println("Error: (host){} != (device){}", cpuPos[i], cudaPos[i]);
            }
        }
    }
    if (passed) {
        std::println("  OK");
    }
    return passed;
}

auto initGL(int* argc, char** argv, bool full_screen) -> void {
    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(1920, 1080);
    glutCreateWindow("CUDA n-body system");

    if (full_screen) {
        glutFullScreen();
    }

    else if (!isGLVersionSupported(2, 0)
             || !areGLExtensionsSupported("GL_ARB_multitexture "
                                          "GL_ARB_vertex_buffer_object")) {
        throw std::runtime_error("Required OpenGL extensions missing.");
    } else {
#if defined(WIN32)
        wglSwapIntervalEXT(0);
#elif defined(LINUX)
        glxSwapIntervalSGI(0);
#endif
    }

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0, 0.0, 0.0, 1.0);

    GLenum error;

    while ((error = glGetError()) != GL_NO_ERROR) {
        std::println(stderr, "initGL: error - {}", reinterpret_cast<const char*>(gluErrorString(error)));
    }
}

auto display(ComputeConfig& compute, InterfaceConfig& interface, CameraConfig& camera) -> void {
    static auto gflops                = 0.f;
    static auto ifps                  = 0.f;
    static auto interactionsPerSecond = 0.f;

    compute.update_simulation(camera);

    interface.display(compute, camera, interactionsPerSecond, gflops, ifps);

    static int fpsCount = 0;
    static int fpsLimit = 5;

    fpsCount++;

    // this displays the frame rate updated every second (independent of frame rate)
    if (fpsCount >= fpsLimit) {
        auto milliseconds = compute.get_milliseconds_passed();

        milliseconds /= static_cast<float>(fpsCount);
        {
            const auto [interactions_per_second, g_flops] = compute.computePerfStats(milliseconds, 1);

            interactionsPerSecond = interactions_per_second;
            gflops                = g_flops;
        }

        ifps = 1000.f / milliseconds;

        const auto fps_str =
            std::format("CUDA N-Body ({} bodies): {:.1f} fps | {:.1f} BIPS | {:.1f} GFLOP/s | {}", compute.num_bodies, ifps, interactionsPerSecond, gflops, compute.fp64_enabled ? "double precision" : "single precision");

        glutSetWindowTitle(fps_str.c_str());
        fpsCount = 0;

        if (compute.paused) {
            fpsLimit = 0;
        } else if (ifps > 1.f) {
            // setting the refresh limit (in number of frames) to be the FPS value obviously refreshes this message every second...
            fpsLimit = static_cast<int>(ifps);
        } else {
            fpsLimit = 1;
        }

        compute.restart_timer();
    }

    glutReportErrors();
}

auto mouse(int button, int state, int x, int y, InterfaceConfig& interface, ControlsConfig& controls, ComputeConfig& compute) -> void {
    if (interface.show_sliders && interface.param_list->is_mouse_over(x, y)) {
        // call list mouse function
        interface.param_list->modify_sliders(x, y, button, state);
        compute.update_params();
    }

    controls.set_state(button, state, x, y);

    glutPostRedisplay();
}

auto motion(int x, int y, InterfaceConfig& interface, ControlsConfig& controls, CameraConfig& camera, ComputeConfig& compute) -> void {
    if (interface.show_sliders) {
        // call parameter list motion function
        if (interface.param_list->Motion(x, y)) {
            compute.update_params();
            glutPostRedisplay();
            return;
        }
    }

    controls.move_camera(camera, x, y);

    glutPostRedisplay();
}

auto key(unsigned char key, [[maybe_unused]] int x, [[maybe_unused]] int y, ComputeConfig& compute, InterfaceConfig& interface, CameraConfig& camera) -> void {
    using enum NBodyConfig;

    switch (key) {
        case ' ':
            compute.pause();
            break;

        case 27:    // escape
        case 'q':
        case 'Q':
            compute.finalize();
            exit(EXIT_SUCCESS);
            break;

        case 13:    // return
            compute.switch_precision();
            break;

        case '`':
            interface.toggle_sliders();
            break;

        case 'g':
        case 'G':
            interface.toggle_interactions();
            break;

        case 'p':
        case 'P':
            interface.cycle_display_mode();
            break;

        case 'c':
        case 'C':
            compute.toggle_cycle_demo();
            break;

        case '[':
            compute.previous_demo(camera);
            break;

        case ']':
            compute.next_demo(camera);
            break;

        case 'd':
        case 'D':
            interface.togle_display();
            break;

        case 'o':
        case 'O':
            compute.active_params.print();
            break;

        case '1':
            compute.reset<NBODY_CONFIG_SHELL>();
            break;

        case '2':
            compute.reset<NBODY_CONFIG_RANDOM>();
            break;

        case '3':
            compute.reset<NBODY_CONFIG_EXPAND>();
            break;
    }

    glutPostRedisplay();
}

///
/// @brief  Describes the various outcomes after parsing the command-line arguments.
///
enum class Status {
    OK = 0,             // Proceed with the rest of the program
    CleanShutDown,      // Everything was fine but exit the program normally
    InvalidArguments    // Something went wrong parsing the command-line arguments!
};

struct Options {
    bool                  fullscreen = false;
    bool                  fp64       = false;
    bool                  hostmem    = false;
    bool                  benchmark  = false;
    std::size_t           numbodies  = 0;
    int                   device     = -1;
    std::size_t           numdevices = 0;
    bool                  compare    = false;
    bool                  qatest     = false;
    bool                  cpu        = false;
    std::filesystem::path tipsy;
    std::size_t           i          = 0;
    std::size_t           block_size = 0;
};

auto parse_args(int argc, char** argv) -> std::pair<Status, Options> {
    auto options = Options{};

    auto app = CLI::App{"The CUDA NBody sample demo.", "cuda-nbody"};

    auto display_version = false;

    app.add_flag("--fullscreen", options.fullscreen, "Run n-body simulation in fullscreen mode");
    app.add_flag("--fp64", options.fp64, "Use double precision floating point values for simulation");
    app.add_flag("--hostmem", options.hostmem, "Stores simulation data in host memory");
    app.add_flag("--benchmark", options.benchmark, "Run benchmark to measure performance");
    app.add_option("--numbodies", options.numbodies, "Number of bodies (>= 1) to run in simulation")->check(CLI::Range(std::size_t{1u}, std::numeric_limits<std::size_t>::max()));
    const auto device_opt = app.add_option("--device", options.device, "The CUDA device to use")->check(CLI::Range(0, std::numeric_limits<int>::max()));
    app.add_option("--numdevices", options.numdevices, "Number of CUDA devices (> 0) to use for simulation")->check(CLI::Range(std::size_t{1u}, std::numeric_limits<std::size_t>::max()))->excludes(device_opt);
    app.add_flag("--compare", options.compare, "Compares simulation results running once on the default GPU and once on the CPU");
    app.add_flag("--qatest", options.qatest, "Runs a QA test");
    app.add_flag("--cpu", options.cpu, "Run n-body simulation on the CPU");
    app.add_option("--tipsy", options.tipsy, "Load a tipsy model file for simulation")->check(CLI::ExistingFile);
    app.add_option("-i,--iterations", options.i, "Number of iterations to run in the benchmark")->default_val(10);
    app.add_option("--blockSize", options.block_size, "The CUDA kernel block size")->default_val(256);

    // cppcheck-suppress unmatchedSuppression
    // cppcheck-suppress passedByValue
    auto error = [&](std::string_view message) {
        std::println(stderr,
                     "-------------------------------------------\n"
                     "CRITICAL ERROR:\n"
                     "{}\n"
                     "-------------------------------------------\n",
                     message);

        std::println(stderr, "{}", app.help());

        return std::pair(Status::InvalidArguments, std::move(options));
    };

    try {
        app.parse(argc, argv);

        if (display_version) {
            std::println("cuda-nbody: {}", git_commit_id);
            return std::pair(Status::CleanShutDown, std::move(options));
        }
    } catch (const CLI::CallForHelp&) {
        std::println("{}", app.help());

        return std::pair(Status::CleanShutDown, std::move(options));
    } catch (const CLI::ParseError& e) { return error(e.what()); }

    std::println(R"(Run " nbody - benchmark[-numbodies = <numBodies>] " to measure performance)");
    std::println("{}", app.help());

    return std::pair(Status::OK, std::move(options));
}

// get the parameter list of a lambda (with some minor fixes): https://stackoverflow.com/a/70954691
template <typename T> struct Signature;
template <typename C, typename... Args> struct Signature<void (C::*)(Args...) const> {
    using type = typename std::tuple<Args...>;
};
template <typename C> struct Signature<void (C::*)() const> {
    using type = void;
};

template <typename F>
concept is_functor = std::is_class_v<std::decay_t<F>> && requires(F&& t) { &std::decay_t<F>::operator(); };

template <is_functor T> auto arguments(T&& t) -> Signature<decltype(&std::decay_t<T>::operator())>::type;

template <auto GLUTFunction, typename T> struct RegisterCallback;

template <auto GLUTFunction> struct RegisterCallback<GLUTFunction, void> {
    template <is_functor F> static auto callback(void* f_data) -> void {
        const auto* obj = static_cast<F*>(f_data);

        return obj->operator()();
    }

    template <typename F> static auto register_callback(F& func) -> void { GLUTFunction(callback<F>, static_cast<void*>(&func)); }
};

template <auto GLUTFunction, typename... Args> struct RegisterCallback<GLUTFunction, std::tuple<Args...>> {
    template <is_functor F> static auto callback(Args... args, void* f_data) -> void {
        const auto* obj = static_cast<F*>(f_data);

        return obj->operator()(args...);
    }

    template <typename F> static auto register_callback(F& func) -> void { GLUTFunction(callback<F>, static_cast<void*>(&func)); }
};

template <auto GLUTFunction, typename F> auto register_callback(F& func) -> void {
    using Args = std::decay_t<decltype(arguments(func))>;
    RegisterCallback<GLUTFunction, Args>::register_callback(func);
}

auto execute_graphics_loop(ComputeConfig& compute, InterfaceConfig& interface, CameraConfig& camera, ControlsConfig& controls) -> void {
    auto display_ = [&]() { display(compute, interface, camera); };

    auto reshape_ = [](int w, int h) {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(60.0, static_cast<float>(w) / static_cast<float>(h), 0.1, 1000.0);

        glMatrixMode(GL_MODELVIEW);
        glViewport(0, 0, w, h);
    };

    auto mouse_   = [&](int button, int state, int x, int y) { mouse(button, state, x, y, interface, controls, compute); };
    auto motion_  = [&](int x, int y) { motion(x, y, interface, controls, camera, compute); };
    auto key_     = [&](unsigned char k, int x, int y) { key(k, x, y, compute, interface, camera); };
    auto special_ = [&](int key, int x, int y) {
        interface.param_list->Special(key, x, y);
        glutPostRedisplay();
    };
    auto idle_ = []() { glutPostRedisplay(); };

    static_assert(std::is_same_v<decltype(arguments(display_)), void>);
    static_assert(std::is_same_v<decltype(arguments(reshape_)), std::tuple<int, int>>);

    register_callback<glutDisplayFuncUcall>(display_);
    register_callback<glutReshapeFuncUcall>(reshape_);
    register_callback<glutMotionFuncUcall>(motion_);
    register_callback<glutMouseFuncUcall>(mouse_);
    register_callback<glutKeyboardFuncUcall>(key_);
    register_callback<glutSpecialFuncUcall>(special_);
    register_callback<glutIdleFuncUcall>(idle_);

    if (!compute.use_cpu) {
        checkCudaErrors(cudaEventRecord(compute.start_event, 0));
    }

    glutMainLoop();
}

template <std::floating_point T> auto run_program(int nb_iterations, ComputeConfig& compute, InterfaceConfig& interface, CameraConfig& camera, ControlsConfig& controls) -> bool {
    if (compute.benchmark) {
        if (nb_iterations <= 0) {
            nb_iterations = 10;
        }

        if (compute.use_cpu) {
            NBodyDemo<BodySystemCPU<T>>::runBenchmark(nb_iterations, compute);
        } else {
            NBodyDemo<BodySystemCUDA<T>>::runBenchmark(nb_iterations, compute);
        }

        return true;
    }
    if (compute.compare_to_cpu) {
        return compare_results(compute.num_bodies, NBodyDemo<BodySystemCUDA<T>>::get_impl());
    }

    assert(interface.param_list != nullptr);

    execute_graphics_loop(compute, interface, camera, controls);

    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    try {
        // parse the command-line arguments
        const auto program_state = parse_args(argc, argv);

        const auto program_status = program_state.first;

        // check the arguments were valid and if we should continue
        if (Status::InvalidArguments == program_status) {
            // treat invalid arguments as an error and exit the program
            return 1;
        }
        if (Status::CleanShutDown == program_status) {
            // shut down the program normally if required (e.g. if --help was requested)
            return 0;
        }

        const auto cmd_options = program_state.second;

        bool bTestResults = true;

#if defined(__linux__)
        setenv("DISPLAY", ":0", 0);
#endif

        std::println("NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

        const auto full_screen = cmd_options.fullscreen;

        std::println("> {} mode", full_screen ? "Fullscreen" : "Windowed");

        auto show_sliders = !full_screen;

        auto compare_to_cpu = cmd_options.compare || cmd_options.qatest;

        auto use_host_mem = cmd_options.hostmem;

        auto numDevsRequested = 1;

        if (cmd_options.numdevices > 0) {
            numDevsRequested = static_cast<int>(cmd_options.numdevices);
            std::println("number of CUDA devices  = {}", numDevsRequested);
        }

        auto customGPU = false;
        {
            int numDevsAvailable = 0;
            cudaGetDeviceCount(&numDevsAvailable);

            if (numDevsAvailable < numDevsRequested) {
                throw std::invalid_argument(std::format("Error: only {} Devices available, {} requested.", numDevsAvailable, numDevsRequested));
            }
        }

        auto useP2P = true;    // this is always optimal to use P2P path when available

        if (numDevsRequested > 1) {
            // If user did not explicitly request host memory to be used, we default to P2P.
            // We fallback to host memory, if any of GPUs does not support P2P.
            auto allGPUsSupportP2P = true;
            if (!use_host_mem) {
                // Enable P2P only in one direction, as every peer will access gpu0
                for (int i = 1; i < numDevsRequested; ++i) {
                    int canAccessPeer;
                    checkCudaErrors(cudaDeviceCanAccessPeer(&canAccessPeer, i, 0));

                    if (canAccessPeer != 1) {
                        allGPUsSupportP2P = false;
                    }
                }

                if (!allGPUsSupportP2P) {
                    use_host_mem = true;
                    useP2P       = false;
                }
            }
        }

        std::println("> Simulation data stored in {} memory", use_host_mem ? "system" : "video");
        std::println("> {} precision floating point simulation", cmd_options.fp64 ? "Double" : "Single");
        std::println("> {} Devices used for simulation", numDevsRequested);

        if (cmd_options.cpu) {
            use_host_mem   = true;
            compare_to_cpu = false;

#ifdef OPENMP
            std::println("> Simulation with CPU using OpenMP");
#else
            std::println("> Simulation with CPU");
#endif
        }

        auto tipsy_file = cmd_options.tipsy;
        auto cycle_demo = tipsy_file.empty();
        show_sliders    = tipsy_file.empty();

        auto compute = ComputeConfig{
            .paused                = false,
            .fp64_enabled          = cmd_options.fp64,
            .cycle_demo            = cycle_demo,
            .active_demo           = 0,
            .use_cpu               = cmd_options.cpu,
            .num_bodies            = 16384,
            .double_supported      = cmd_options.cpu,
            .flops_per_interaction = cmd_options.fp64 ? 30 : 20,
            .compare_to_cpu        = compare_to_cpu,
            .benchmark             = cmd_options.benchmark,
            .use_host_mem          = use_host_mem,
            .active_params         = ComputeConfig::demoParams[0],
            .host_mem_sync_event   = cudaEvent_t{},
            .start_event           = cudaEvent_t{},
            .stop_event            = cudaEvent_t{}};

        const auto enable_graphics = !compute.benchmark && !compute.compare_to_cpu;

        // Initialize GL and GLUT if necessary
        if (enable_graphics) {
            initGL(&argc, argv, full_screen);
        }

        int devID = 0;

        cudaDeviceProp props{};
        if (!cmd_options.cpu) {
            if (cmd_options.device != -1) {
                customGPU = true;
            }

            // If the command-line has a device number specified, use it
            if (customGPU) {
                devID = cmd_options.device;
                assert(devID >= 0);

                const auto new_dev_ID = gpuDeviceInit(devID);

                if (new_dev_ID < 0) {
                    throw std::invalid_argument(std::format("Could not use custom CUDA device: {}", devID));
                }

                devID = new_dev_ID;

            } else {
                // Otherwise pick the device with highest Gflops/s
                devID = gpuGetMaxGflopsDeviceId();
                checkCudaErrors(cudaSetDevice(devID));
                int major = 0, minor = 0;
                checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
                checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
                std::println(R"(GPU Device {}: "{}" with compute capability {}.{}\n)", devID, _ConvertSMVer2ArchName(major, minor), major, minor);
            }

            checkCudaErrors(cudaGetDevice(&devID));
            checkCudaErrors(cudaGetDeviceProperties(&props, devID));

            compute.double_supported = true;

            // Initialize devices
            assert(!(customGPU && (numDevsRequested > 1)));

            if (customGPU || numDevsRequested == 1) {
                cudaDeviceProp props1;
                checkCudaErrors(cudaGetDeviceProperties(&props1, devID));
                std::println("> Compute {}.{} CUDA device: [{}]", props1.major, props1.minor, props1.name);
                // CC 1.2 and earlier do not support double precision
                if (props1.major * 10 + props1.minor <= 12) {
                    compute.double_supported = false;
                }

            } else {
                for (int i = 0; i < numDevsRequested; i++) {
                    cudaDeviceProp props2;
                    checkCudaErrors(cudaGetDeviceProperties(&props2, i));

                    std::println("> Compute {}.{} CUDA device: [{}]", props2.major, props2.minor, props2.name);

                    if (compute.use_host_mem) {
                        if (!props2.canMapHostMemory) {
                            throw std::invalid_argument(std::format("Device {} cannot map host memory!", i));
                        }

                        if (numDevsRequested > 1) {
                            checkCudaErrors(cudaSetDevice(i));
                        }

                        checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
                    }

                    // CC 1.2 and earlier do not support double precision
                    if (props2.major * 10 + props2.minor <= 12) {
                        compute.double_supported = false;
                    }
                }
            }

            if (compute.fp64_enabled && !compute.double_supported) {
                throw std::invalid_argument("One or more of the requested devices does not support double precision floating-point");
            }
        }

        auto numIterations = static_cast<int>(cmd_options.i);
        auto blockSize     = static_cast<int>(cmd_options.block_size);

        // default number of bodies is #SMs * 4 * CTA size
        if (cmd_options.cpu) {
#ifdef OPENMP
            compute.num_bodies = 8192;
#else
            compute.num_bodies = 4096;
#endif
        } else if (numDevsRequested == 1) {
            compute.num_bodies = compute.compare_to_cpu ? 4096 : blockSize * 4 * props.multiProcessorCount;
        } else {
            compute.num_bodies = 0;
            for (int i = 0; i < numDevsRequested; i++) {
                cudaDeviceProp props1;
                checkCudaErrors(cudaGetDeviceProperties(&props1, i));
                compute.num_bodies += blockSize * (props1.major >= 2 ? 4 : 1) * props1.multiProcessorCount;
            }
        }

        if (cmd_options.numbodies != 0u) {
            compute.num_bodies = static_cast<int>(cmd_options.numbodies);

            assert(compute.num_bodies >= 1);

            if (compute.num_bodies % blockSize) {
                int newNumBodies = ((compute.num_bodies / blockSize) + 1) * blockSize;
                std::println(R"(Warning: "number of bodies" specified {} is not a multiple of {}.)", compute.num_bodies, blockSize);
                std::println("Rounding up to the nearest multiple: {}.", newNumBodies);
                compute.num_bodies = newNumBodies;
            } else {
                std::println("number of bodies = {}", compute.num_bodies);
            }
        }

        if (compute.num_bodies <= 1024) {
            compute.active_params.m_clusterScale  = 1.52f;
            compute.active_params.m_velocityScale = 2.f;
        } else if (compute.num_bodies <= 2048) {
            compute.active_params.m_clusterScale  = 1.56f;
            compute.active_params.m_velocityScale = 2.64f;
        } else if (compute.num_bodies <= 4096) {
            compute.active_params.m_clusterScale  = 1.68f;
            compute.active_params.m_velocityScale = 2.98f;
        } else if (compute.num_bodies <= 8192) {
            compute.active_params.m_clusterScale  = 1.98f;
            compute.active_params.m_velocityScale = 2.9f;
        } else if (compute.num_bodies <= 16384) {
            compute.active_params.m_clusterScale  = 1.54f;
            compute.active_params.m_velocityScale = 8.f;
        } else if (compute.num_bodies <= 32768) {
            compute.active_params.m_clusterScale  = 1.44f;
            compute.active_params.m_velocityScale = 11.f;
        }

        auto interface = InterfaceConfig{
            .display_enabled      = true,
            .show_sliders         = show_sliders,
            .param_list           = enable_graphics ? compute.active_params.create_sliders() : nullptr,
            .full_screen          = full_screen,
            .display_interactions = false,
            .display_mode         = ParticleRenderer::PARTICLE_SPRITES_COLOR};

        auto camera = CameraConfig{.translation_lag = {0.f, -2.f, -150.f}, .translation = {0.f, -2.f, -150.f}, .rotation = {0.f, 0.f, 0.f}};

        auto controls = ControlsConfig{.button_state = 0, .old_x = 0, .old_y = 0};

        using enum NBodyConfig;

        // Create the demo -- either double (fp64) or float (fp32, default)
        // implementation
        NBodyDemo<BodySystemCPU<float>>::Create(tipsy_file);
        NBodyDemo<BodySystemCPU<float>>::init(numDevsRequested, blockSize, useP2P, devID, compute);
        NBodyDemo<BodySystemCPU<float>>::reset(compute, NBODY_CONFIG_SHELL);

        NBodyDemo<BodySystemCUDA<float>>::Create(tipsy_file);
        NBodyDemo<BodySystemCUDA<float>>::init(numDevsRequested, blockSize, useP2P, devID, compute);
        NBodyDemo<BodySystemCUDA<float>>::reset(compute, NBODY_CONFIG_SHELL);

        if (compute.double_supported) {
            NBodyDemo<BodySystemCPU<double>>::Create(tipsy_file);
            NBodyDemo<BodySystemCPU<double>>::init(numDevsRequested, blockSize, useP2P, devID, compute);
            NBodyDemo<BodySystemCPU<double>>::reset(compute, NBODY_CONFIG_SHELL);

            NBodyDemo<BodySystemCUDA<double>>::Create(tipsy_file);
            NBodyDemo<BodySystemCUDA<double>>::init(numDevsRequested, blockSize, useP2P, devID, compute);
            NBodyDemo<BodySystemCUDA<double>>::reset(compute, NBODY_CONFIG_SHELL);
        }

        if (compute.fp64_enabled) {
            bTestResults = run_program<double>(numIterations, compute, interface, camera, controls);
        } else {
            bTestResults = run_program<float>(numIterations, compute, interface, camera, controls);
        }

        compute.finalize();

        if (!bTestResults) {
            return 1;
        }
        return 0;
    } catch (const std::invalid_argument& e) {
        std::println(stderr, "ERROR: {}", e.what());
        return 1;
    } catch (const std::bad_alloc&) {
        std::println(stderr, "ERROR: Unable to allocate memory!");
        return 3;
    } catch (const std::exception& e) {
        std::println(stderr, "ERROR: ", e.what());
        return 2;
    } catch (...) {
        std::println("ERROR: An unknown error occurred! Please inform your local developer!");
        return 4;
    }
}
