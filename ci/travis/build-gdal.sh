#!/usr/bin/env sh

REBUILD_GDAL=false

# Build GDAL2
if [ ! "$(ls -A ${HOME}/.libs/include/gdal.h)" ] || [ "$REBUILD_GDAL" = true ]; then
  wget -q -O gdal.tar.gz "http://download.osgeo.org/gdal/2.4.1/gdal-2.4.1.tar.gz" > /dev/null
  tar -xzf gdal.tar.gz
  cd gdal-2.4.1
  ./configure --prefix=${HOME}/.libs --with-static-proj4=/usr --with-sqlite3=no --with-python=no --with-pg=no --with-grass=no --with-jasper=/usr --with-curl=/usr --with-jpeg=internal --with-png=internal --disable-shared --enable-static --silent
  make -j6 > /dev/null
  make install
  cd ..
  printf 'GDAL has been built.\n'
else 
  printf 'GDAL will not be built (%s/.libs/include/gdal.h found).\n' "$HOME"
fi
