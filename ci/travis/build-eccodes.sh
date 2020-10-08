#!/usr/bin/env sh

REBUILD_ECCODES=false
ECCODES_VERSION=2.18.0

# Build ecCodes
if [ ! "$(ls -A ${HOME}/.libs/include/eccodes.h)" ] || [ "$REBUILD_ECCODES" = true ]; then
  wget -q -O eccodes.tar.gz "https://confluence.ecmwf.int/download/attachments/45757960/eccodes-${ECCODES_VERSION}-Source.tar.gz" > /dev/null
  tar -xzf eccodes.tar.gz
  cd eccodes-${ECCODES_VERSION}-Source
  mkdir bld
  cd bld
  cmake .. -DCMAKE_INSTALL_PREFIX=${HOME}/.libs -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DENABLE_JPG=ON -DENABLE_PYTHON=OFF -DENABLE_FORTRAN=OFF -DCMAKE_PREFIX_PATH="${HOME}/.libs" > /dev/null
  make -j$(nproc) > /dev/null
  make install > /dev/null
  cd ..
  cd ..
  printf 'ecCodes has been built.\n'
else 
  printf 'ecCodes will not be built (%s/.libs/include/eccodes.h found).\n' "$HOME"
fi
