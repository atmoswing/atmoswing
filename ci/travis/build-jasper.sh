#!/usr/bin/env sh

REBUILD_JASPER=false
JASPER_VERSION=2.0.16

# Build Jasper
if [ ! "$(ls -A ${HOME}/.libs/include/jasper/jasper.h)" ] || [ "$REBUILD_JASPER" = true ]; then
  wget -q -O jasper.tar.gz "https://github.com/mdadams/jasper/archive/version-${JASPER_VERSION}.tar.gz" > /dev/null
  tar -xzf jasper.tar.gz
  cd jasper-version-${JASPER_VERSION}
  mkdir bld
  cd bld
  cmake .. -DCMAKE_INSTALL_PREFIX=${HOME}/.libs -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${HOME}/.libs" > /dev/null
  make -j$(nproc) > /dev/null
  make install
  cd ..
  printf 'Jasper has been built.\n'
else 
  printf 'Jasper will not be built (%s/.libs/include/jasper/jasper.h found).\n' "$HOME"
fi
