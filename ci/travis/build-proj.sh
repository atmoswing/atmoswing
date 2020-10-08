#!/usr/bin/env sh

REBUILD_PROJ=false
PROJ_VERSION=7.1.1

# Build PROJ
if [ ! "$(ls -A ${HOME}/.libs/include/proj.h)" ] || [ "$REBUILD_PROJ" = true ]; then
  wget -q -O proj.tar.gz "https://download.osgeo.org/proj/proj-${PROJ_VERSION}.tar.gz" > /dev/null
  tar -xzf proj.tar.gz
  cd proj-${PROJ_VERSION}
  ./configure --prefix=${HOME}/.libs --silent
  make -j$(nproc) > /dev/null
  make install
  cd ..
  printf 'PROJ has been built.\n'
else 
  printf 'PROJ will not be built (%s/.libs/include/proj.h found).\n' "$HOME"
fi
