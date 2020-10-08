#!/usr/bin/env sh

CMAKE_VERSION=3.18.4

sudo apt purge --auto-remove -y cmake
sudo find /usr/local -type f -name "cmake" -exec rm -rf {} \;
wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh > /dev/null
sudo mkdir /opt/cmake
sudo sh cmake-${CMAKE_VERSION}-Linux-x86_64.sh --skip-license --prefix=/opt/cmake > /dev/null
sudo ln -s /opt/cmake/bin/cmake /usr/bin/cmake
sudo ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
