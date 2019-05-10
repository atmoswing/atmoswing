#!/bin/bash

REBUILD_GDAL=true

# Build GDAL2
if [ ! "$(ls -A ${HOME}/.libs/include/gdal.h)" ] || [ "$REBUILD_GDAL" = true ]; then
  wget -q -O gdal.tar.gz "http://download.osgeo.org/gdal/3.0.0/gdal-3.0.0.tar.gz" > /dev/null
  tar -xzf gdal.tar.gz
  cd gdal-2.4.1
  ./configure --prefix=${HOME}/.libs --with-proj=/usr/local --with-sqlite3=no --with-python=no --with-pg=no --with-grass=no --with-jasper=/usr --with-curl=/usr --with-jpeg=internal --with-png=internal --silent
  make -j6 > /dev/null
  make install
  cd ..
  printf 'GDAL has been built.\n'
else 
  printf 'GDAL will not be built (%s/.libs/include/gdal.h found).\n' "$HOME"
fi
