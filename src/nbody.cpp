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
#include "nbody/helper_cuda.hpp"
#include "nbody/paramgl.hpp"
#include "nbody/render_particles.hpp"

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

constexpr auto computePerfStats(float nb_bodies, int flops_per_interaction, float milliseconds, int iterations) -> std::array<double, 2> {
    // double precision uses intrinsic operation followed by refinement, resulting in higher operation count per interaction.
    // Note: Astrophysicists use 38 flops per interaction no matter what, based on "historical precedent", but they are using FLOP/s as a measure of "science throughput".
    // We are using it as a measure of hardware throughput.  They should really use interactions/s...
    const auto interactionsPerSecond = (nb_bodies * nb_bodies * 1e-9f) * iterations * (1000.0f / milliseconds);

    const auto gflops = interactionsPerSecond * static_cast<float>(flops_per_interaction);

    return {interactionsPerSecond, gflops};
}

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

    auto print() const -> void {
        std::println("{{ {}, {}, {}, {}, {}, {}, {}, {}, {} }},", m_timestep, m_clusterScale, m_velocityScale, m_softening, m_damping, m_pointSize, camera_origin[0], camera_origin[1], camera_origin[2]);
    }

    auto create_sliders() -> std::unique_ptr<ParamListGL> {
        // create a new parameter list
        auto paramlist = std::make_unique<ParamListGL>("sliders");
        paramlist->SetBarColorInner(0.8f, 0.8f, 0.0f);

        // add some parameters to the list

        // Point Size
        paramlist->AddParam(std::make_unique<Param<float>>("Point Size", m_pointSize, 0.001f, 10.0f, 0.01f, &m_pointSize));

        // Velocity Damping
        paramlist->AddParam(std::make_unique<Param<float>>("Velocity Damping", m_damping, 0.5f, 1.0f, .0001f, &m_damping));
        // Softening Factor
        paramlist->AddParam(std::make_unique<Param<float>>("Softening Factor", m_softening, 0.001f, 1.0f, .0001f, &m_softening));
        // Time step size
        paramlist->AddParam(std::make_unique<Param<float>>("Time Step", m_timestep, 0.0f, 1.0f, .0001f, &m_timestep));
        // Cluster scale (only affects starting configuration
        paramlist->AddParam(std::make_unique<Param<float>>("Cluster Scale", m_clusterScale, 0.0f, 10.0f, 0.01f, &m_clusterScale));

        // Velocity scale (only affects starting configuration)
        paramlist->AddParam(std::make_unique<Param<float>>("Velocity Scale", m_velocityScale, 0.0f, 1000.0f, 0.1f, &m_velocityScale));

        return paramlist;
    }
};

constexpr static auto demoParams = std::array{
    NBodyParams{0.016f, 1.54f, 8.0f, 0.1f, 1.0f, 1.0f, 0, -2, -100},
    NBodyParams{0.016f, 0.68f, 20.0f, 0.1f, 1.0f, 0.8f, 0, -2, -30},
    NBodyParams{0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    NBodyParams{0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0.07f, 0, 0, -1.5f},
    NBodyParams{0.0019f, 0.32f, 276.0f, 1.0f, 1.0f, 0.07f, 0, 0, -5},
    NBodyParams{0.0016f, 0.32f, 272.0f, 0.145f, 1.0f, 0.08f, 0, 0, -5},
    NBodyParams{0.016000f, 6.040000f, 0.000000f, 1.000000f, 1.000000f, 0.760000f, 0, 0, -50}};

constexpr static auto numDemos = demoParams.size();

constexpr static auto demoTime = 10000.0f;    // ms

using Clock        = std::chrono::steady_clock;
using TimePoint    = std::chrono::time_point<Clock>;
using MilliSeconds = std::chrono::duration<float, std::milli>;

template <std::floating_point T> class NBodyDemo;

struct CameraConfig {
    std::array<float, 3> translation_lag;
    std::array<float, 3> translation;
    std::array<float, 3> rotation;

    auto reset(const std::array<float, 3>& origin) noexcept -> void { translation = translation_lag = origin; }
};

struct ComputeConfig;

void selectDemo(ComputeConfig& compute, CameraConfig& camera);

struct ComputeConfig {
    bool        paused;
    bool        fp64_enabled;
    bool        cycle_demo;
    int         active_demo;
    bool        use_cpu;
    int         num_bodies;
    bool        double_supported;
    int         flops_per_interaction;
    bool        compare_to_cpu;
    bool        benchmark;
    bool        use_host_mem;
    NBodyParams active_params;
    cudaEvent_t host_mem_sync_event;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    template <std::floating_point T_new, std::floating_point T_old> auto switch_demo_precision() -> void {
        static_assert(!std::is_same_v<T_new, T_old>);

        cudaDeviceSynchronize();

        fp64_enabled          = !fp64_enabled;
        flops_per_interaction = fp64_enabled ? 30 : 20;

        const auto nb_bodies_4 = static_cast<std::size_t>(num_bodies * 4);

        auto oldPos = std::vector<T_old>(nb_bodies_4);
        auto oldVel = std::vector<T_old>(nb_bodies_4);

        NBodyDemo<T_old>::getArrays(oldPos, oldVel);

        // convert float to double
        auto newPos = std::vector<T_new>(nb_bodies_4);
        auto newVel = std::vector<T_new>(nb_bodies_4);

        for (int i = 0; i < nb_bodies_4; i++) {
            newPos[i] = static_cast<T_new>(oldPos[i]);
            newVel[i] = static_cast<T_new>(oldVel[i]);
        }

        NBodyDemo<T_new>::setArrays(newPos, newVel, *this);

        cudaDeviceSynchronize();
    }

    template <std::floating_point T> auto run_benchmark(int iterations, BodySystem<T>& nbody) -> void {
        // once without timing to prime the device
        if (!use_cpu) {
            nbody.update(active_params.m_timestep);
        }

        auto milliseconds = 0.f;
        auto start        = TimePoint{};

        if (use_cpu) {
            start = Clock::now();
        } else {
            checkCudaErrors(cudaEventRecord(start_event, 0));
        }

        for (int i = 0; i < iterations; ++i) {
            nbody.update(active_params.m_timestep);
        }

        if (use_cpu) {
            milliseconds = MilliSeconds{Clock::now() - start}.count();
        } else {
            checkCudaErrors(cudaEventRecord(stop_event, 0));
            checkCudaErrors(cudaEventSynchronize(stop_event));
            checkCudaErrors(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        }

        const auto [interactionsPerSecond, gflops] = computePerfStats(static_cast<float>(num_bodies), flops_per_interaction, milliseconds, iterations);

        std::println("%d bodies, total time for %d iterations: %.3f ms", num_bodies, iterations, milliseconds);
        std::println("= %.3f billion interactions per second", interactionsPerSecond);
        std::println("= %.3f %s-precision GFLOP/s at %d flops per interaction", gflops, (sizeof(T) > 4) ? "double" : "single", flops_per_interaction);
    }

    auto select_demo() -> void {
        assert(active_demo < numDemos);

        active_params = demoParams[active_demo];
    }

    auto finalize() noexcept -> void;

    auto pause() noexcept -> void { paused = !paused; }

    auto switch_precision() -> void {
        if (double_supported) {
            if (fp64_enabled) {
                switch_demo_precision<float, double>();
                std::println("> Double precision floating point simulation");
            } else {
                switch_demo_precision<double, float>();
                std::println("> Single precision floating point simulation");
            }
        }
    }

    auto toggle_cycle_demo() -> void {
        cycle_demo = !cycle_demo;
        std::println("Cycle Demo Parameters: {}\n", cycle_demo ? "ON" : "OFF");
    }

    auto previous_demo(CameraConfig& camera) -> void {
        active_demo = (active_demo == 0) ? numDemos - 1 : (active_demo - 1) % numDemos;
        selectDemo(*this, camera);
    }

    auto next_demo(CameraConfig& camera) -> void {
        active_demo = (active_demo + 1) % numDemos;
        selectDemo(*this, camera);
    }
};

struct ControlsConfig {
    int button_state;
    int old_x;
    int old_y;
};

struct InterfaceConfig {
    bool                          display_enabled;
    bool                          show_sliders;
    std::unique_ptr<ParamListGL>  param_list;
    bool                          full_screen;
    bool                          display_interactions;
    ParticleRenderer::DisplayMode display_mode;

    auto toggle_sliders() noexcept -> void { show_sliders = !show_sliders; }
    auto toggle_interactions() noexcept -> void { display_interactions = !display_interactions; }
    auto cycle_display_mode() noexcept -> void { display_mode = (ParticleRenderer::DisplayMode)((display_mode + 1) % ParticleRenderer::PARTICLE_NUM_MODES); }
    auto togle_display() noexcept -> void { display_enabled = !display_enabled; }
};

using std::ranges::copy;

template <std::floating_point T> class NBodyDemo {
 public:
    static void Create(const std::filesystem::path& tipsy_file) { m_singleton = std::make_unique<NBodyDemo>(tipsy_file); }
    static void Destroy() noexcept { m_singleton.reset(); }

    static void init(int numDevices, int block_size, bool use_p2p, int devID, ComputeConfig& compute) { m_singleton->_init(numDevices, block_size, use_p2p, devID, compute); }

    static void reset(ComputeConfig& compute, NBodyConfig config) { m_singleton->_reset(compute, config); }

    static void selectDemo(ComputeConfig& compute, CameraConfig& camera) { m_singleton->_selectDemo(compute, camera); }

    static bool compareResults(int num_bodies) { return m_singleton->_compareResults(num_bodies); }

    static void runBenchmark(int iterations, ComputeConfig& compute) { compute.run_benchmark(iterations, *(m_singleton->m_nbody)); }

    static void updateParams(const NBodyParams& active_params) {
        m_singleton->m_nbody->setSoftening(active_params.m_softening);
        m_singleton->m_nbody->setDamping(active_params.m_damping);
    }

    static void updateSimulation(float dt) { m_singleton->m_nbody->update(dt); }

    static void display(const ComputeConfig& compute, ParticleRenderer::DisplayMode display_mode) {
        m_singleton->m_renderer->setSpriteSize(compute.active_params.m_pointSize);

        if (compute.use_host_mem) {
            // This event sync is required because we are rendering from the host memory that CUDA is writing.
            // If we don't wait until CUDA is done updating it, we will render partially updated data, resulting in a jerky frame rate.
            if (!compute.use_cpu) {
                cudaEventSynchronize(compute.host_mem_sync_event);
            }

            m_singleton->m_renderer->setPositions(m_singleton->m_nbody->getArray(BODYSYSTEM_POSITION), m_singleton->m_nbody->getNumBodies());
        } else {
            m_singleton->m_renderer->setPBO(m_singleton->m_nbody->getCurrentReadBuffer(), m_singleton->m_nbody->getNumBodies(), std::is_same_v<T, double>);
        }

        // display particles
        m_singleton->m_renderer->display(display_mode);
    }

    static void getArrays(std::vector<T>& pos, std::vector<T>& vel) {
        T* _pos = m_singleton->m_nbody->getArray(BODYSYSTEM_POSITION);
        T* _vel = m_singleton->m_nbody->getArray(BODYSYSTEM_VELOCITY);
        copy(_pos, _pos + m_singleton->m_nbody->getNumBodies() * 4, pos.begin());
        copy(_vel, _vel + m_singleton->m_nbody->getNumBodies() * 4, vel.begin());
    }

    static void setArrays(const std::vector<T>& pos, const std::vector<T>& vel, const ComputeConfig& compute) {
        if (pos.data() != m_singleton->m_hPos.data()) {
            copy(pos, m_singleton->m_hPos.begin());
        }

        if (vel.data() != m_singleton->m_hVel.data()) {
            copy(vel, m_singleton->m_hVel.begin());
        }

        m_singleton->m_nbody->setArray(BODYSYSTEM_POSITION, m_singleton->m_hPos.data());
        m_singleton->m_nbody->setArray(BODYSYSTEM_VELOCITY, m_singleton->m_hVel.data());

        if (!compute.benchmark && !compute.use_cpu && !compute.compare_to_cpu) {
            m_singleton->_resetRenderer(compute.active_params.m_pointSize);
        }
    }

    static auto get_demo_time() -> float { return MilliSeconds{Clock::now() - m_singleton->demo_reset_time_}.count(); }

    static auto get_milliseconds_passed() -> float {
        const auto now           = Clock::now();
        const auto milliseconds  = MilliSeconds{Clock::now() - m_singleton->reset_time_}.count();
        m_singleton->reset_time_ = now;

        return milliseconds;
    }

    NBodyDemo(std::filesystem::path tipsy_file) : tipsy_file_(std::move(tipsy_file)) {}

 private:
    static std::unique_ptr<NBodyDemo> m_singleton;

    BodySystem<T>*                     m_nbody = nullptr;
    std::unique_ptr<BodySystemCUDA<T>> m_nbodyCuda;
    std::unique_ptr<BodySystemCPU<T>>  m_nbodyCpu;

    std::unique_ptr<ParticleRenderer> m_renderer;

    std::vector<T>     m_hPos;
    std::vector<T>     m_hVel;
    std::vector<float> m_hColor;

    TimePoint demo_reset_time_;

    TimePoint reset_time_;

    std::filesystem::path tipsy_file_;

 private:
    void _init(int numDevices, int block_size, bool use_p2p, int devID, ComputeConfig& compute) {
        if (compute.use_cpu) {
            m_nbodyCpu = std::make_unique<BodySystemCPU<T>>(compute.num_bodies);
            m_nbody    = m_nbodyCpu.get();
        } else {
            const auto use_pbo = !(compute.benchmark || compute.compare_to_cpu || compute.use_host_mem);
            m_nbodyCuda        = std::make_unique<BodySystemCUDA<T>>(compute.num_bodies, numDevices, block_size, use_pbo, compute.use_host_mem, use_p2p, devID);
            m_nbody            = m_nbodyCuda.get();
        }

        const auto nb_bodies_4 = compute.num_bodies * 4;

        // allocate host memory
        m_hPos.resize(nb_bodies_4);
        m_hVel.resize(nb_bodies_4);
        m_hColor.resize(nb_bodies_4);

        m_nbody->setSoftening(compute.active_params.m_softening);
        m_nbody->setDamping(compute.active_params.m_damping);

        if (compute.use_cpu) {
            reset_time_ = Clock::now();
        } else {
            checkCudaErrors(cudaEventCreate(&compute.start_event));
            checkCudaErrors(cudaEventCreate(&compute.stop_event));
            checkCudaErrors(cudaEventCreate(&compute.host_mem_sync_event));
        }

        if (!compute.benchmark && !compute.compare_to_cpu) {
            m_renderer = std::make_unique<ParticleRenderer>();
            _resetRenderer(compute.active_params.m_pointSize);
        }

        demo_reset_time_ = Clock::now();
    }

    void _reset(ComputeConfig& compute, NBodyConfig config) {
        if (tipsy_file_.empty()) {
            randomizeBodies(config, m_hPos.data(), m_hVel.data(), m_hColor.data(), compute.active_params.m_clusterScale, compute.active_params.m_velocityScale, compute.num_bodies, true);
            setArrays(m_hPos, m_hVel, compute);
        } else {
            m_nbody->loadTipsyFile(tipsy_file_);
            compute.num_bodies = m_nbody->getNumBodies();
        }
    }

    void _resetRenderer(float point_size) {
        const auto colour = std::is_same_v<T, double> ? std::array{0.4f, 0.8f, 0.1f, 1.0f} : std::array{1.0f, 0.6f, 0.3f, 1.0f};

        m_renderer->setBaseColor(colour);
        m_renderer->setColors(m_hColor.data(), m_nbody->getNumBodies());
        m_renderer->setSpriteSize(point_size);
    }

    void _selectDemo(ComputeConfig& compute, CameraConfig& camera) {
        compute.select_demo();

        camera.reset(compute.active_params.camera_origin);

        _reset(compute, NBODY_CONFIG_SHELL);

        demo_reset_time_ = Clock::now();
    }

    bool _compareResults(int num_bodies) {
        assert(m_nbodyCuda);

        bool passed = true;

        m_nbody->update(0.001f);

        {
            m_nbodyCpu = std::make_unique<BodySystemCPU<T>>(num_bodies);

            m_nbodyCpu->setArray(BODYSYSTEM_POSITION, m_hPos.data());
            m_nbodyCpu->setArray(BODYSYSTEM_VELOCITY, m_hVel.data());

            m_nbodyCpu->update(0.001f);

            T* cudaPos = m_nbodyCuda->getArray(BODYSYSTEM_POSITION);
            T* cpuPos  = m_nbodyCpu->getArray(BODYSYSTEM_POSITION);

            T tolerance = 0.0005f;

            for (int i = 0; i < num_bodies; i++) {
                if (std::abs(cpuPos[i] - cudaPos[i]) > tolerance) {
                    passed = false;
                    std::println("Error: (host)%f != (device)%f", cpuPos[i], cudaPos[i]);
                }
            }
        }
        if (passed) {
            std::println("  OK");
        }
        return passed;
    }
};

auto ComputeConfig::finalize() noexcept -> void {
    if (!use_cpu) {
        checkCudaErrors(cudaEventDestroy(start_event));
        checkCudaErrors(cudaEventDestroy(stop_event));
        checkCudaErrors(cudaEventDestroy(host_mem_sync_event));
    }

    NBodyDemo<float>::Destroy();

    if (double_supported)
        NBodyDemo<double>::Destroy();
}

template <> std::unique_ptr<NBodyDemo<double>> NBodyDemo<double>::m_singleton = nullptr;
template <> std::unique_ptr<NBodyDemo<float>>  NBodyDemo<float>::m_singleton  = nullptr;

void initGL(int* argc, char** argv, bool full_screen) {
    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA
    // interop.
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

void selectDemo(ComputeConfig& compute, CameraConfig& camera) {
    if (compute.fp64_enabled) {
        NBodyDemo<double>::selectDemo(compute, camera);
    } else {
        NBodyDemo<float>::selectDemo(compute, camera);
    }
}

void updateSimulation(bool fp64_enabled, float dt) {
    if (fp64_enabled) {
        NBodyDemo<double>::updateSimulation(dt);
    } else {
        NBodyDemo<float>::updateSimulation(dt);
    }
}

void displayNBodySystem(const ComputeConfig& compute, ParticleRenderer::DisplayMode display_mode) {
    if (compute.fp64_enabled) {
        NBodyDemo<double>::display(compute, display_mode);
    } else {
        NBodyDemo<float>::display(compute, display_mode);
    }
}

void display(ComputeConfig& compute, InterfaceConfig& interface, CameraConfig& camera) {
    static double gflops                = 0;
    static double ifps                  = 0;
    static double interactionsPerSecond = 0;

    // update the simulation
    if (!compute.paused) {
        const auto demo_time = compute.fp64_enabled ? NBodyDemo<double>::get_demo_time() : NBodyDemo<float>::get_demo_time();

        if (compute.cycle_demo && (demo_time > demoTime)) {
            compute.active_demo = (compute.active_demo + 1) % numDemos;
            selectDemo(compute, camera);
        }

        updateSimulation(compute.fp64_enabled, compute.active_params.m_timestep);

        if (!compute.use_cpu) {
            cudaEventRecord(compute.host_mem_sync_event, 0);    // insert an event to wait on before rendering
        }
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (interface.display_enabled) {
        constexpr static auto inertia = 0.1f;

        // view transform
        {
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            static auto camera_rot_lag = std::array{0.f, 0.f, 0.f};

            for (int c = 0; c < 3; ++c) {
                camera.translation_lag[c] += (camera.translation[c] - camera.translation_lag[c]) * inertia;
                camera_rot_lag[c] += (camera.rotation[c] - camera_rot_lag[c]) * inertia;
            }

            glTranslatef(camera.translation_lag[0], camera.translation_lag[1], camera.translation_lag[2]);
            glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
            glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);
        }

        displayNBodySystem(compute, interface.display_mode);

        // display user interface
        if (interface.show_sliders) {
            glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);    // invert color
            glEnable(GL_BLEND);
            interface.param_list->Render(0, 0);
            glDisable(GL_BLEND);
        }

        if (interface.full_screen) {
            beginWinCoords();
            constexpr static auto& msg0 = "some_temp_device_name";
            char                   msg1[256], msg2[256];
            // char deviceName[100];

            if (interface.display_interactions) {
                sprintf(msg1, "%0.2f billion interactions per second", interactionsPerSecond);
            } else {
                sprintf(msg1, "%0.2f GFLOP/s", gflops);
            }

            // sprintf(msg0, "%s", deviceName);
            sprintf(msg2, "%0.2f FPS [%s | %d bodies]", ifps, compute.fp64_enabled ? "double precision" : "single precision", compute.num_bodies);

            glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);    // invert color
            glEnable(GL_BLEND);
            glColor3f(0.46f, 0.73f, 0.0f);
            glPrint(80, glutGet(GLUT_WINDOW_HEIGHT) - 122, msg0, GLUT_BITMAP_TIMES_ROMAN_24);
            glColor3f(1.0f, 1.0f, 1.0f);
            glPrint(80, glutGet(GLUT_WINDOW_HEIGHT) - 96, msg2, GLUT_BITMAP_TIMES_ROMAN_24);
            glColor3f(1.0f, 1.0f, 1.0f);
            glPrint(80, glutGet(GLUT_WINDOW_HEIGHT) - 70, msg1, GLUT_BITMAP_TIMES_ROMAN_24);
            glDisable(GL_BLEND);

            endWinCoords();
        }

        glutSwapBuffers();
    }

    static int fpsCount = 0;
    static int fpsLimit = 5;

    fpsCount++;

    // this displays the frame rate updated every second (independent of frame rate)
    if (fpsCount >= fpsLimit) {
        char fps[256];

        float milliseconds = 1;

        // stop timer
        if (compute.use_cpu) {
            milliseconds = compute.fp64_enabled ? NBodyDemo<double>::get_milliseconds_passed() : NBodyDemo<float>::get_milliseconds_passed();
        } else {
            checkCudaErrors(cudaEventRecord(compute.stop_event, 0));
            checkCudaErrors(cudaEventSynchronize(compute.stop_event));
            checkCudaErrors(cudaEventElapsedTime(&milliseconds, compute.start_event, compute.stop_event));
        }

        milliseconds /= (float)fpsCount;
        {
            const auto [interactions_per_second, g_flops] = computePerfStats(static_cast<float>(compute.num_bodies), compute.flops_per_interaction, milliseconds, 1);

            interactionsPerSecond = interactions_per_second;
            gflops                = g_flops;
        }

        ifps = 1.f / (milliseconds / 1000.f);
        sprintf(fps,
                "CUDA N-Body (%d bodies): "
                "%0.1f fps | %0.1f BIPS | %0.1f GFLOP/s | %s",
                compute.num_bodies,
                ifps,
                interactionsPerSecond,
                gflops,
                compute.fp64_enabled ? "double precision" : "single precision");

        glutSetWindowTitle(fps);
        fpsCount = 0;
        fpsLimit = (ifps > 1.f) ? (int)ifps : 1;

        if (compute.paused) {
            fpsLimit = 0;
        }

        // restart timer
        if (!compute.use_cpu) {
            checkCudaErrors(cudaEventRecord(compute.start_event, 0));
        }
    }

    glutReportErrors();
}

void updateParams(bool fp64_enabled, const NBodyParams& active_params) {
    if (fp64_enabled) {
        NBodyDemo<double>::updateParams(active_params);
    } else {
        NBodyDemo<float>::updateParams(active_params);
    }
}

void mouse(int button, int state, int x, int y, InterfaceConfig& interface, ControlsConfig& controls, bool fp64_enabled, const NBodyParams& active_params) {
    if (interface.show_sliders) {
        // call list mouse function
        if (interface.param_list->Mouse(x, y, button, state)) {
            updateParams(fp64_enabled, active_params);
        }
    }

    if (state == GLUT_DOWN) {
        controls.button_state |= 1 << button;
    } else if (state == GLUT_UP) {
        controls.button_state = 0;
    }

    const auto mods = glutGetModifiers();

    if (mods & GLUT_ACTIVE_SHIFT) {
        controls.button_state = 2;
    } else if (mods & GLUT_ACTIVE_CTRL) {
        controls.button_state = 3;
    }

    controls.old_x = x;
    controls.old_y = y;

    glutPostRedisplay();
}

void motion(int x, int y, InterfaceConfig& interface, ControlsConfig& controls, CameraConfig& camera, bool fp64_enabled, const NBodyParams& active_params) {
    if (interface.show_sliders) {
        // call parameter list motion function
        if (interface.param_list->Motion(x, y)) {
            updateParams(fp64_enabled, active_params);
            glutPostRedisplay();
            return;
        }
    }

    const auto dx = static_cast<float>(x - controls.old_x);
    const auto dy = static_cast<float>(y - controls.old_y);

    if (controls.button_state == 3) {
        // left+middle = zoom
        camera.translation[2] += (dy / 100.0f) * 0.5f * std::abs(camera.translation[2]);
    } else if (controls.button_state & 2) {
        // middle = translate
        camera.translation[0] += dx / 100.0f;
        camera.translation[1] -= dy / 100.0f;
    } else if (controls.button_state & 1) {
        // left = rotate
        camera.rotation[0] += dy / 5.0f;
        camera.rotation[1] += dx / 5.0f;
    }

    controls.old_x = x;
    controls.old_y = y;
    glutPostRedisplay();
}

void key(unsigned char key, [[maybe_unused]] int x, [[maybe_unused]] int y, ComputeConfig& compute, InterfaceConfig& interface, CameraConfig& camera) {
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
            if (compute.fp64_enabled) {
                NBodyDemo<double>::reset(compute, NBODY_CONFIG_SHELL);
            } else {
                NBodyDemo<float>::reset(compute, NBODY_CONFIG_SHELL);
            }

            break;

        case '2':
            if (compute.fp64_enabled) {
                NBodyDemo<double>::reset(compute, NBODY_CONFIG_RANDOM);
            } else {
                NBodyDemo<float>::reset(compute, NBODY_CONFIG_RANDOM);
            }

            break;

        case '3':
            if (compute.fp64_enabled) {
                NBodyDemo<double>::reset(compute, NBODY_CONFIG_EXPAND);
            } else {
                NBodyDemo<float>::reset(compute, NBODY_CONFIG_EXPAND);
            }

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

    auto mouse_   = [&](int button, int state, int x, int y) { mouse(button, state, x, y, interface, controls, compute.fp64_enabled, compute.active_params); };
    auto motion_  = [&](int x, int y) { motion(x, y, interface, controls, camera, compute.fp64_enabled, compute.active_params); };
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

        NBodyDemo<T>::runBenchmark(nb_iterations, compute);

        return true;
    }
    if (compute.compare_to_cpu) {
        return NBodyDemo<T>::compareResults(compute.num_bodies);
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

        auto show_sliders = !full_screen;

        auto compare_to_cpu = cmd_options.compare || cmd_options.qatest;

        auto use_host_mem = cmd_options.hostmem;

        auto numDevsRequested = 1;

        if (cmd_options.numdevices > 0) {
            numDevsRequested = static_cast<int>(cmd_options.numdevices);
            std::println("number of CUDA devices  = {}", numDevsRequested);
        }

        int  numDevsAvailable = 0;
        bool customGPU        = false;
        cudaGetDeviceCount(&numDevsAvailable);

        if (numDevsAvailable < numDevsRequested) {
            throw std::invalid_argument(std::format("Error: only {} Devices available, {} requested.", numDevsAvailable, numDevsRequested));
        }

        auto useP2P = true;    // this is always optimal to use P2P path when available

        if (numDevsRequested > 1) {
            // If user did not explicitly request host memory to be used, we default to P2P.
            // We fallback to host memory, if any of GPUs does not support P2P.
            bool allGPUsSupportP2P = true;
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

        std::println("> {} mode", full_screen ? "Fullscreen" : "Windowed");
        std::println("> Simulation data stored in {} memory", use_host_mem ? "system" : "video");
        std::println("> {} precision floating point simulation", cmd_options.fp64 ? "Double" : "Single");
        std::println("> {} Devices used for simulation", numDevsRequested);

        int            devID = 0;
        cudaDeviceProp props{};

        if (cmd_options.cpu) {
            use_host_mem   = true;
            compare_to_cpu = false;

#ifdef OPENMP
            std::println("> Simulation with CPU using OpenMP");
#else
            std::println("> Simulation with CPU");
#endif
        }

        auto tipsy_file = std::filesystem::path{};
        auto cycle_demo = true;

        if (!cmd_options.tipsy.empty()) {
            tipsy_file   = cmd_options.tipsy;
            cycle_demo   = false;
            show_sliders = false;
        }

        auto param_list = std::unique_ptr<ParamListGL>{};

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
            .active_params         = demoParams[0],
            .host_mem_sync_event   = cudaEvent_t{},
            .start_event           = cudaEvent_t{},
            .stop_event            = cudaEvent_t{}};

        // Initialize GL and GLUT if necessary
        if (!compute.benchmark && !compute.compare_to_cpu) {
            initGL(&argc, argv, full_screen);
            param_list = compute.active_params.create_sliders();
        }

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

        auto camera = CameraConfig{.translation_lag = {0.f, -2.f, -150.f}, .translation = {0.f, -2.f, -150.f}, .rotation = {0.f, 0.f, 0.f}};

        auto controls = ControlsConfig{.button_state = 0, .old_x = 0, .old_y = 0};

        auto interface = InterfaceConfig{
            .display_enabled      = true,
            .show_sliders         = show_sliders,
            .param_list           = std::move(param_list),
            .full_screen          = full_screen,
            .display_interactions = false,
            .display_mode         = ParticleRenderer::PARTICLE_SPRITES_COLOR};

        // Create the demo -- either double (fp64) or float (fp32, default)
        // implementation
        NBodyDemo<float>::Create(tipsy_file);

        NBodyDemo<float>::init(numDevsRequested, blockSize, useP2P, devID, compute);
        NBodyDemo<float>::reset(compute, NBODY_CONFIG_SHELL);

        if (compute.double_supported) {
            NBodyDemo<double>::Create(tipsy_file);
            NBodyDemo<double>::init(numDevsRequested, blockSize, useP2P, devID, compute);
            NBodyDemo<double>::reset(compute, NBODY_CONFIG_SHELL);
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
