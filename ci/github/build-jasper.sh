#!/usr/bin/env sh

JASPER_VERSION=2.0.16

# Build Jasper
wget -q -O jasper.tar.gz "https://github.com/mdadams/jasper/archive/version-${JASPER_VERSION}.tar.gz"
tar -xzf jasper.tar.gz
cd jasper-version-${JASPER_VERSION}
mkdir bld
cd bld
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j $(nproc)
sudo make install
cd ..
printf 'Jasper has been built.\n'
