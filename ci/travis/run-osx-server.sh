#!/usr/bin/env sh

export OPENSSL_ROOT_DIR="/usr/local/opt/openssl"

cmake CMakeLists.txt -DBUILD_OPTIMIZER=1 -DBUILD_FORECASTER=1 -DBUILD_DOWNSCALER=1 -DBUILD_VIEWER=0 -DCREATE_INSTALLER=ON -DUSE_GUI=0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$HOME/.libs
make -j6

echo "-----------"
pwd
echo "-----------"
find /Users/travis/ -name atmoswing-tests
echo "-----------"
ctest -V
