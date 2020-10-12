#!/usr/bin/env bash

cmake CMakeLists.txt -DBUILD_OPTIMIZER=1 -DBUILD_FORECASTER=1 -DBUILD_DOWNSCALER=1 -DBUILD_VIEWER=0 -DUSE_GUI=0 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CODECOV=1 -DGDAL_PATH=$HOME/.libs -DCMAKE_PREFIX_PATH=$HOME/.libs > /dev/null
echo "Building target"
make -j$(nproc) atmoswing-coverage > /dev/null
echo "Preparing coverage data"
lcov --directory . --capture --output-file coverage.info &> /dev/null
lcov --remove coverage.info "/usr/*" "*/_deps/*" "*/tests/*" --output-file coverage.info
lcov --list coverage.info
echo "Sending to Codecov"
bash <(curl -s https://codecov.io/bash) -f coverage.info || echo "Codecov did not collect coverage reports"
