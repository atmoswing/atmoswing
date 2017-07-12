#!/usr/bin/env sh

cmake CMakeLists.txt -DBUILD_OPTIMIZER=1 -DBUILD_FORECASTER=1 -DBUILD_VIEWER=1 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CODECOV=1 -DGDAL_ROOT=$HOME/.libs/gdal
make -j$(nproc) atmoswing-coverage > /dev/null
lcov --directory . --capture --output-file coverage.info --quiet
lcov --remove coverage.info '*/tests/*' 'bin/*' '*/libs/*' '*/.libs/*' '*/include/*' '/usr/*' --output-file coverage.info
lcov --list coverage.info --quiet
curl -s https://codecov.io/bash || echo "Codecov did not collect coverage reports"
