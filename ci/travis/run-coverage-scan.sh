#!/usr/bin/env bash

cmake CMakeLists.txt -DBUILD_OPTIMIZER=1 -DBUILD_FORECASTER=1 -DBUILD_DOWNSCALER=1 -DBUILD_VIEWER=0 -DUSE_GUI=0 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CODECOV=1 -DGDAL_ROOT=$HOME/.libs -DCMAKE_PREFIX_PATH=$HOME/.libs
make -j$(nproc) atmoswing-coverage > /dev/null
lcov --directory . --capture --output-file coverage.info &> /dev/null
lcov --remove coverage.info '/usr/*' '*/external/*' '*/tests/*' --output-file coverage.info --quiet
lcov --list coverage.info
bash <(curl -s https://codecov.io/bash) || echo "Codecov did not collect coverage reports" 
