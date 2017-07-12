#!/usr/bin/env sh

cmake CMakeLists.txt -DBUILD_OPTIMIZER=1 -DBUILD_FORECASTER=1 -DBUILD_VIEWER=1 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGDAL_ROOT=$HOME/.libs/gdal
make -j$(nproc)
ctest -V