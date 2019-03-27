#!/usr/bin/env sh

REBUILD_ECCODES=false

# Build ecCodes
if [ ! "$(ls -A ${HOME}/.libs/include/eccodes.h)" ] || [ "$REBUILD_ECCODES" = true ]; then
  wget -q -O eccodes.tar.gz "https://confluence.ecmwf.int/download/attachments/45757960/eccodes-2.12.0-Source.tar.gz" > /dev/null
  tar -xzf eccodes.tar.gz
  cd eccodes-2.12.0-Source
  mkdir bld
  cd bld
  cmake .. -DCMAKE_INSTALL_PREFIX=${HOME}/.libs -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DENABLE_JPG=ON -DENABLE_PYTHON=OFF -DENABLE_FORTRAN=OFF -DCMAKE_PREFIX_PATH=${HOME}/.libs > /dev/null
  make -j6 > /dev/null
  make install
  cd ..
  cd ..
  printf 'ecCodes has been built.\n'
else 
  printf 'ecCodes will not be built (%s/.libs/include/eccodes.h found).\n' "$HOME"
fi
