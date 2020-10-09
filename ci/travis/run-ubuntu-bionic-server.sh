#!/usr/bin/env sh

cmake CMakeLists.txt -DBUILD_OPTIMIZER=1 -DBUILD_FORECASTER=1 -DBUILD_DOWNSCALER=1 -DBUILD_VIEWER=0 -DCREATE_INSTALLER=1 -DUSE_GUI=0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$HOME/.libs
make -j$(nproc)

cd tests
./atmoswing-tests
