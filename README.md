# cuda-nbody

This is the cuda-nbody project.

Clone recursively to obtain `vcpkg` and the project dependencies. e.g.

```
$ git clone --recursive git@github.com:c4x-discovery/cuda-nbody.git
```
# Additional dependencies
- A recent version of CMake i.e 3.21 or later. VS 2019 has native support for 3.20 so it will need to be explicitly installed.
- A recent compiler supporting C++17 or later.
- `ninja-build` on Linux, if not change the `"generator"` from `"Ninja"` to `"Unix Makefiles"`
- Cppcheck on Windows if you want to use the `"cppcheck"` preset.

See the [BUILDING](BUILDING.md) document.

# Building and installing

Tested building in following environments:
  - Visual Studio 2022

If your IDE has good native support for CMake Presets it should work fine (Visual Studio and VS Code do).
Alternatively, on Linux, the provided `build_script.sh` can be used, e.g. for the `dev-unix` preset:
```
$ ./build_script.sh dev-unix --run-tests
```

See the [BUILDING](BUILDING.md) document for some more information.

# Licensing

<!--
Please go to https://choosealicense.com/ and choose a license that fits your
needs. GNU GPLv3 is a pretty nice option ;-)
-->
