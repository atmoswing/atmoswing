#!/usr/bin/env sh

REBUILD_GDAL=false
GDAL_VERSION=3.1.3

# Build GDAL2
if [ ! "$(ls -A ${HOME}/.libs/include/gdal.h)" ] || [ "$REBUILD_GDAL" = true ]; then
  wget -q -O gdal.tar.gz "http://download.osgeo.org/gdal/${GDAL_VERSION}/gdal-${GDAL_VERSION}.tar.gz" > /dev/null
  tar -xzf gdal.tar.gz
  cd gdal-${GDAL_VERSION}
  ./configure --prefix=${HOME}/.libs --with-proj=${HOME}/.libs --with-sqlite3=no --with-python=no --with-pg=no --with-grass=no --with-jasper=${HOME}/.libs --with-curl=/usr --with-jpeg=internal --with-png=internal --disable-shared --enable-static --silent
  make -j$(nproc) > /dev/null
  make install
  cd ..
  printf 'GDAL has been built.\n'
else 
  printf 'GDAL will not be built (%s/.libs/include/gdal.h found).\n' "$HOME"
fi
