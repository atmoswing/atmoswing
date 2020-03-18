#!/usr/bin/env sh

sudo apt purge --auto-remove -y cmake
sudo find /usr/local -type f -name "cmake" -exec rm -rf {} \;
wget https://github.com/Kitware/CMake/releases/download/v3.17.0-rc3/cmake-3.17.0-rc3-Linux-x86_64.sh > /dev/null
sudo mkdir /opt/cmake
sudo sh cmake-3.17.0-rc3-Linux-x86_64.sh --skip-license --prefix=/opt/cmake > /dev/null
sudo ln -s /opt/cmake/bin/cmake /usr/bin/cmake
sudo ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
