#!/bin/bash -u

TOOLS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

find "${TOOLS_DIR}/../src/" "${TOOLS_DIR}/../include/" -iname *.hpp -o -iname *.cpp | xargs bash -c 'clang-format -i "$@"' _
