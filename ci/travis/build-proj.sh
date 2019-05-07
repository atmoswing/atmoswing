#!/usr/bin/env sh

REBUILD_PROJ=false

# Build PROJ4
if [ ! "$(ls -A ${HOME}/.libs/include/proj.h)" ] || [ "$REBUILD_PROJ" = true ]; then
  wget -q -O proj.tar.gz "http://download.osgeo.org/proj/proj-6.0.0.tar.gz" > /dev/null
  tar -xzf proj.tar.gz
  ls
  cd proj-6.0.0
  ./configure --prefix=${HOME}/.libs --silent
  make -j6 > /dev/null
  make install
  cd ..
  printf 'PROJ has been built.\n'
else 
  printf 'PROJ will not be built (%s/.libs/include/proj.h found).\n' "$HOME"
fi
