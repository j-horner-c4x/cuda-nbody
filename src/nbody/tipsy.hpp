#pragma once

#include <array>
#include <filesystem>
#include <fstream>
#include <ios>
#include <print>
#include <vector>

using Vector3D = std::array<float, 3>;

struct GasParticle {
    float    mass;
    Vector3D pos;
    Vector3D vel;
    float    rho;
    float    temp;
    float    hsmooth;
    float    metals;
    float    phi;
};

struct DarkParticle {
    float    mass;
    Vector3D pos;
    Vector3D vel;
    float    eps;
    int      phi;
};

struct StarParticle {
    float    mass;
    Vector3D pos;
    Vector3D vel;
    float    metals;
    float    tform;
    float    eps;
    int      phi;
};

struct Dump {
    double time;
    int    nbodies;
    int    ndim;
    int    nsph;
    int    ndark;
    int    nstar;
};

template <typename real4> void read_tipsy_file(std::vector<real4>& bodyPositions, std::vector<real4>& bodyVelocities, const std::filesystem::path& fileName) {
    // Read in our custom version of the tipsy file format written by Jeroen Bedorf.
    // Most important change is that we store particle id on the location where previously the potential was stored.

    std::println("Trying to read file: {}", fileName.string());

    std::ifstream inputFile(fileName, std::ios::in | std::ios::binary);

    if (!inputFile.is_open()) {
        throw std::runtime_error("Can't open input file");
    }

    auto read_data = [&](auto& data) { inputFile.read(reinterpret_cast<char*>(&data), sizeof(data)); };

    Dump h;
    read_data(h);

    int   idummy;
    real4 positions;
    real4 velocity;

    // Read tipsy header
    auto NTotal = h.nbodies;
    auto NFirst = h.ndark;

    DarkParticle d;
    StarParticle s;

    for (int i = 0; i < NTotal; i++) {
        if (i < NFirst) {
            read_data(d);
            velocity.w  = d.eps;
            positions.w = d.mass;
            positions.x = d.pos[0];
            positions.y = d.pos[1];
            positions.z = d.pos[2];
            velocity.x  = d.vel[0];
            velocity.y  = d.vel[1];
            velocity.z  = d.vel[2];
            idummy      = d.phi;
        } else {
            read_data(s);
            velocity.w  = s.eps;
            positions.w = s.mass;
            positions.x = s.pos[0];
            positions.y = s.pos[1];
            positions.z = s.pos[2];
            velocity.x  = s.vel[0];
            velocity.y  = s.vel[1];
            velocity.z  = s.vel[2];
            idummy      = s.phi;
        }

        bodyPositions.push_back(positions);
        bodyVelocities.push_back(velocity);
    }

    // round up to a multiple of 256 bodies since our kernel only supports that...
    int newTotal = NTotal;

    if (NTotal % 256) {
        newTotal = ((NTotal / 256) + 1) * 256;
    }

    for (int i = NTotal; i < newTotal; i++) {
        positions.w = positions.x = positions.y = positions.z = 0;
        velocity.x = velocity.y = velocity.z = 0;
        bodyPositions.push_back(positions);
        bodyVelocities.push_back(velocity);
    }

    assert(bodyPositions.size() == newTotal);
    assert(bodyVelocities.size() == newTotal);

    std::println("Read {} bodies", newTotal);
}
