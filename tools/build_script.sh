#!/bin/bash -u

PREV_DIR=${PWD}

CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/../"

CMAKE=cmake

if [[ -z "${INSTANCE_TYPE+x}" ]]; then
    # INSTANCE_TYPE is not defined: we are not on an AWS EC2 instance
    ON_AWS_EC2=false
else
    # INSTANCE_TYPE is defined: we are on an AWS EC2 instance
    ON_AWS_EC2=true
fi

print_usage() {
cat <<- EOF
    ** build_script.sh **

    This script will compile cuda-nbody with a given set of options.
    <preset name>           : CMake Preset to build. Required positional argument. Run "$ cmake --list-presets" to see available options.
    [-h|--help]             : displays this message
    [-t|--run-tests]        : runs CTest after build and install

    By default cuda-nbody will be compiled in Debug mode with gcc.

    $(cd "${CODE_DIR}" && ${CMAKE} --list-presets)
EOF
}

PRESET=""
PRESET_FOUND=false
RUN_TESTS=false
for arg in "$@"
do
    case $arg in
        -h|--help)
            print_usage
            exit 0
            ;;
        -t|--run-tests)
            RUN_TESTS=true
            ;;
        *)
            if ${PRESET_FOUND} ; then
                echo "ERROR: Unrecognised command line argument - ${arg}"
                print_usage
                exit 1
            else
                PRESET=$arg
                PRESET_FOUND=true
            fi
            ;;
    esac
done

if [[ "${ON_AWS_EC2}" = true ]]; then
    # put build artefacts in instance-type specific folders
    BUILD_DIR=${CODE_DIR}/build/${PRESET}/${INSTANCE_TYPE}
    INSTALL_DIR=${CODE_DIR}/build/install/${PRESET}/${INSTANCE_TYPE}
else
    # directory for all uninteresting build artifacts to go e.g. Makefiles, library files, etc.
    BUILD_DIR=${CODE_DIR}/build/${PRESET}
    # directory for all generated executables e.g. cuda-nbody, unit tests etc.
    INSTALL_DIR=${CODE_DIR}/build/install/${PRESET}
fi

echo
echo -e "\033[33m----------------------------------------------------------------------------"
echo -e "\t\t\tCompiling - preset: ${PRESET}"
echo -e "----------------------------------------------------------------------------\033[0m"
echo

cd "${CODE_DIR}" || exit

git fetch --tags --all --quiet --force

# check if either directory doesn't exist, if so we need to run CMake to generate the build system
if [ ! -d "${BUILD_DIR}" ]; then
    ${CMAKE} --preset "${PRESET}" "-L"
fi

${CMAKE} --build "${BUILD_DIR}" --target install -j

if ${RUN_TESTS} ; then
    cd "${BUILD_DIR}" && ctest --output-on-failure -j
fi

cd "${PREV_DIR}" || exit
