# cuda-nbody

This is is a port of the CUDA NBody sample code:
https://github.com/NVIDIA/cuda-samples/blob/master/Samples/5_Domain_Specific/nbody/README.md

Goals:
- Isolate the code from the other samples and utilities and build in a self-contained manner.
- Manage (non-CUDA) dependencies with `vcpkg`.
- Setup project in a manner that is largely inspired by: https://github.com/friendlyanon/cmake-init
- Remove legacy cruft and generally simplify the codebase.

Clone recursively to obtain `vcpkg` and the project dependencies. e.g.

```
$ git clone --recursive git@github.com:j-horner/cuda-nbody.git
```
# Additional dependencies
- A recent version of CMake i.e 3.31.6 or later.
- A recent compiler supporting C++23 or later.
- `ninja-build` on Linux, if not change the `"generator"` from `"Ninja"` to `"Unix Makefiles"`
- Cppcheck on Windows if you want to use the `"cppcheck"` preset.

See the [BUILDING](BUILDING.md) document.

# Building and installing

Tested building in following environments:
  - Visual Studio 2022

If your IDE has good native support for CMake Presets it should work fine.
Alternatively, on Linux, the provided `build_script.sh` can be used (mostly a wwrapper around standard `cmake` commands), e.g. for the `dev-unix` preset:
```
$ ./build_script.sh dev-unix --run-tests
```

See the [BUILDING](BUILDING.md) document for some more information.

# Licensing

Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
