# Hacking

Here is some wisdom to help you build and test this project as a developer and potential contributor.

## Developer mode

Build system targets that are only useful for developers of this project are hidden if the `cuda-nbody_DEVELOPER_MODE` option is disabled.
Enabling this option makes tests and other developer targets and options available.
Not enabling this option means that you are a consumer of this project and thus you have no need for these targets and options.

Developer mode is always set to on in CI workflows.

### Presets

This project makes use of [presets][1] to simplify the process of configuring the project.
As a developer, you are recommended to always have the [latest CMake version][2] installed to make use of the latest Quality-of-Life additions.

You have a few options to pass `cuda-nbody_DEVELOPER_MODE` to the configure command, but this project prefers to use presets.

As a developer, you should create a `CMakeUserPresets.json` file at the root of
the project:

```json
{
  "version": 1,
  "cmakeMinimumRequired": {
    "major": 3,
    "minor": 31,
    "patch": 6
  },
  "configurePresets": [
    {
      "name": "dev",
      "inherits": ["dev-mode", "<os>-ci"]
    }
  ]
}
```

You should replace `<os>` in your newly created presets file with the name of the operating system you have, which may be `win64` or `unix`.
You can see what these correspond to in the [`CMakePresets.json`](CMakePresets.json) file.

`CMakeUserPresets.json` is also the perfect place in which you can put all sorts of things that you would otherwise want to pass to the configure command in the terminal.

### Configure, build and test

If you followed the above instructions, then you can configure, build and test the project respectively with the following commands from the project root on Windows:

```sh
cmake --preset=dev
cmake --build build/dev --config Release
cd build/dev && ctest -C Release
```

And here is the same on a Unix based system (Linux, macOS):

```sh
cmake --preset=dev
cmake --build build/dev
cd build/dev && ctest
```

[1]: https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html
[2]: https://cmake.org/download/
