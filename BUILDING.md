# Building with CMake

## Building with command-line

Copy `CMakeUser_presets_example.json` to `CMakeUserPresets.json`.
Choose a preset, listed in either [`CMakePresets.json`](CMakePresets.json) or [`CMakePresetsUser.json`](CMakePresetsUser.json), then run the provided [`build_script.sh`](build_script.sh)

Examples:

```
$ ./build_script.sh ubuntu-ci
$ ./build_script.sh dev-unix --run-tests
```

To list available presets, run `cmake --list-presets` or `./build_script.sh --help`.
Alternatively CMake can be invoked directly. See [`build_script.sh`](build_script.sh).

## Visual Studio

Open the repository folder.
Everything should just work.
See documentation:
  - [CMake projects with Visual Studio](https://docs.microsoft.com/en-gb/cpp/build/cmake-projects-in-visual-studio?view=msvc-170)
  - [CMake projects with Visual Studio and WSL 2](https://docs.microsoft.com/en-gb/cpp/build/walkthrough-build-debug-wsl2?view=msvc-170)

## VS Code
Ensure appropriate [extensions](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools-extension-pack) are installed.
Open the repository folder.
Everything should just work.
See documentation:
  - [CMake Tools for VS Code](https://github.com/microsoft/vscode-cmake-tools/tree/main/docs#cmake-tools-for-visual-studio-code-documentation)
  - [CMake Presets and VS Code](https://github.com/microsoft/vscode-cmake-tools/blob/main/docs/cmake-presets.md)

## CMake options overview

In the presets, some options in particular are defined:
    - `cuda-nbody_DEVELOPER_MODE`: If true, enables various developer specific features such as coverage and documentation generation.
    - `cuda-nbody_BUILD_TESTING`:  If true, enables unit tests.


Additional compiler flags can be specified in `EXTRA_FLAGS` and they will be appropriately appended to those already defined.
