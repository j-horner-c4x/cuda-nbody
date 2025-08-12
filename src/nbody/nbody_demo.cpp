#include "nbody_demo.hpp"

#include "bodysystemcpu.hpp"
#include "bodysystemcuda.hpp"
#include "camera.hpp"
#include "compute.hpp"
#include "helper_cuda.hpp"
#include "randomise_bodies.hpp"

template <> std::unique_ptr<NBodyDemo<BodySystemCPU<double>>>  NBodyDemo<BodySystemCPU<double>>::m_singleton  = nullptr;
template <> std::unique_ptr<NBodyDemo<BodySystemCPU<float>>>   NBodyDemo<BodySystemCPU<float>>::m_singleton   = nullptr;
template <> std::unique_ptr<NBodyDemo<BodySystemCUDA<double>>> NBodyDemo<BodySystemCUDA<double>>::m_singleton = nullptr;
template <> std::unique_ptr<NBodyDemo<BodySystemCUDA<float>>>  NBodyDemo<BodySystemCUDA<float>>::m_singleton  = nullptr;

template <typename BodySystem> auto NBodyDemo<BodySystem>::Create(const std::filesystem::path& tipsy_file) -> void {
    m_singleton = std::make_unique<NBodyDemo>(tipsy_file);
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::Destroy() noexcept -> void {
    m_singleton.reset();
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::init(int numDevices, int block_size, bool use_p2p, int devID, ComputeConfig& compute) -> void {
    m_singleton->_init(numDevices, block_size, use_p2p, devID, compute);
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::reset(ComputeConfig& compute, NBodyConfig config) -> void {
    m_singleton->_reset(compute, config);
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::selectDemo(ComputeConfig& compute, CameraConfig& camera) -> void {
    m_singleton->_selectDemo(compute, camera);
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::runBenchmark(ComputeConfig& compute) -> void {
    compute.run_benchmark(*(m_singleton->m_nbody));
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::updateParams(const NBodyParams& active_params) -> void {
    m_singleton->m_nbody->setSoftening(active_params.m_softening);
    m_singleton->m_nbody->setDamping(active_params.m_damping);
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::updateSimulation(float dt) -> void {
    m_singleton->m_nbody->update(dt);
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::display(const ComputeConfig& compute, ParticleRenderer::DisplayMode display_mode) -> void {
    m_singleton->m_renderer->setSpriteSize(compute.active_params.m_pointSize);

    if (compute.use_host_mem) {
        // This event sync is required because we are rendering from the host memory that CUDA is writing.
        // If we don't wait until CUDA is done updating it, we will render partially updated data, resulting in a jerky frame rate.
        if (!compute.use_cpu) {
            cudaEventSynchronize(compute.host_mem_sync_event);
        }

        m_singleton->m_renderer->setPositions(m_singleton->m_nbody->get_position());
    } else {
        m_singleton->m_renderer->setPBO(m_singleton->m_nbody->getCurrentReadBuffer(), m_singleton->m_nbody->getNumBodies(), std::is_same_v<PrecisionType, double>);
    }

    // display particles
    m_singleton->m_renderer->display(display_mode);
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::getArrays(std::vector<PrecisionType>& pos, std::vector<PrecisionType>& vel) -> void {
    using std::ranges::copy;

    auto _pos = m_singleton->m_nbody->get_position();
    auto _vel = m_singleton->m_nbody->get_velocity();
    copy(_pos, pos.begin());
    copy(_vel, vel.begin());
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::setArrays(const std::vector<PrecisionType>& pos, const std::vector<PrecisionType>& vel, const ComputeConfig& compute) -> void {
    using std::ranges::copy;

    if (pos.data() != m_singleton->m_hPos.data()) {
        copy(pos, m_singleton->m_hPos.begin());
    }

    if (vel.data() != m_singleton->m_hVel.data()) {
        copy(vel, m_singleton->m_hVel.begin());
    }

    m_singleton->m_nbody->set_position(m_singleton->m_hPos);
    m_singleton->m_nbody->set_velocity(m_singleton->m_hVel);

    if (!compute.benchmark && !compute.use_cpu && !compute.compare_to_cpu) {
        m_singleton->_resetRenderer(compute.active_params.m_pointSize);
    }
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::get_demo_time() -> float {
    return MilliSeconds{Clock::now() - m_singleton->demo_reset_time_}.count();
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::get_milliseconds_passed() -> float {
    const auto now           = Clock::now();
    const auto milliseconds  = MilliSeconds{Clock::now() - m_singleton->reset_time_}.count();
    m_singleton->reset_time_ = now;

    return milliseconds;
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::_init(int numDevices, int block_size, bool use_p2p, int devID, ComputeConfig& compute) -> void {
    if constexpr (BodySystem::use_cpu) {
        m_nbody = std::make_unique<BodySystem>(compute.num_bodies);
    } else {
        const auto use_pbo = !(compute.benchmark || compute.compare_to_cpu || compute.use_host_mem);
        m_nbody            = std::make_unique<BodySystem>(compute.num_bodies, numDevices, block_size, use_pbo, compute.use_host_mem, use_p2p, devID);
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

template <typename BodySystem> auto NBodyDemo<BodySystem>::_reset(ComputeConfig& compute, NBodyConfig config) -> void {
    if (tipsy_file_.empty()) {
        randomise_bodies<BodySystem::Type>(config, m_hPos, m_hVel, m_hColor, compute.active_params.m_clusterScale, compute.active_params.m_velocityScale);
        setArrays(m_hPos, m_hVel, compute);
    } else {
        m_nbody->loadTipsyFile(tipsy_file_);
        compute.num_bodies = m_nbody->getNumBodies();
    }
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::_resetRenderer(float point_size) -> void {
    const auto colour = std::is_same_v<PrecisionType, double> ? std::array{0.4f, 0.8f, 0.1f, 1.0f} : std::array{1.0f, 0.6f, 0.3f, 1.0f};

    m_renderer->setBaseColor(colour);
    m_renderer->setColors(m_hColor.data(), m_nbody->getNumBodies());
    m_renderer->setSpriteSize(point_size);
}

template <typename BodySystem> auto NBodyDemo<BodySystem>::_selectDemo(ComputeConfig& compute, CameraConfig& camera) -> void {
    compute.select_demo();

    camera.reset(compute.active_params.camera_origin);

    _reset(compute, NBodyConfig::NBODY_CONFIG_SHELL);

    demo_reset_time_ = Clock::now();
}

template NBodyDemo<BodySystemCPU<float>>;
template NBodyDemo<BodySystemCPU<double>>;
template NBodyDemo<BodySystemCUDA<float>>;
template NBodyDemo<BodySystemCUDA<double>>;