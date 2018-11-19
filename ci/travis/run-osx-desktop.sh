#!/usr/bin/env sh

export OPENSSL_ROOT_DIR="/usr/local/opt/openssl"

cmake CMakeLists.txt -DBUILD_OPTIMIZER=1 -DBUILD_FORECASTER=1 -DBUILD_DOWNSCALER=1 -DBUILD_VIEWER=1 -DUSE_GUI=1 -DCMAKE_BUILD_TYPE=Release -DGDAL_ROOT=$HOME/.libs -DCMAKE_PREFIX_PATH=$HOME/.libs
make -j6

cd tests
open ./atmoswing-tests.app
