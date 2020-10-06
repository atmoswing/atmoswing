#!/usr/bin/env sh

REBUILD_PROJ=false

# Build PROJ
if [ ! "$(ls -A ${HOME}/.libs/include/proj.h)" ] || [ "$REBUILD_PROJ" = true ]; then
  wget -q -O proj.tar.gz "https://download.osgeo.org/proj/proj-7.0.0.tar.gz" > /dev/null
  tar -xzf proj.tar.gz
  cd proj-7.0.0
  ./configure --prefix=${HOME}/.libs --silent
  make -j$(nproc) > /dev/null
  make install
  cd ..
  printf 'PROJ has been built.\n'
else 
  printf 'PROJ will not be built (%s/.libs/include/proj.h found).\n' "$HOME"
fi
