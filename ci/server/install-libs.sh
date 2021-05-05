#!/usr/bin/env bash

PREFIX=${HOME}/libs
TMP=${HOME}/tmp-build

WX_VERSION=3.1.4
CURL_VERSION=7.65.1
TIFF_VERSION=4.2.0
PROJ_VERSION=8.0.0
ZLIB_VERSION=1.2.11
HDF_VERSION=1.12.0
HDF_VERSION_PATH=1.12
NETCDF_VERSION=4.8.0
OPENJPG_VERSION=2.3.1
JASPER_VERSION=2.0.16
ECCODES_VERSION=2.12.0

mkdir $TMP

# wxWidgets
cd $TMP
wget -q -O wxwidgets.tar.bz2 "https://github.com/wxWidgets/wxWidgets/releases/download/v${WX_VERSION}/wxWidgets-${WX_VERSION}.tar.bz2"
tar -xjf wxwidgets.tar.bz2
cd wxWidgets-${WX_VERSION}
mkdir wxbase-build
cd wxbase-build
../configure --enable-unicode --disable-monolithic --disable-shared --disable-debug --disable-gui --prefix=$PREFIX
make -j6
make install

# cURL
cd $TMP
wget https://curl.haxx.se/download/curl-${CURL_VERSION}.tar.gz
tar -xzf curl-${CURL_VERSION}.tar.gz
cd curl-${CURL_VERSION}
./configure --prefix=$PREFIX
make -j6
make check
make install

# Tiff
cd $TMP
wget https://download.osgeo.org/libtiff/tiff-${TIFF_VERSION}.tar.gz
tar -xzf tiff-${TIFF_VERSION}.tar.gz
cd tiff-${TIFF_VERSION}
./configure --prefix=$PREFIX
make -j6
make install

# Proj
cd $TMP
wget https://download.osgeo.org/proj/proj-${PROJ_VERSION}.tar.gz
tar -xzf proj-${PROJ_VERSION}.tar.gz
cd proj-${PROJ_VERSION}
TIFF_CFLAGS=-I${PREFIX}/include TIFF_LIBS="-L${PREFIX}/lib -ltiff" ./configure --prefix=$PREFIX
make -j6
make install

# Zlib
cd $TMP
wget https://zlib.net/zlib-${ZLIB_VERSION}.tar.gz
tar -xzf zlib-${ZLIB_VERSION}.tar.gz
cd zlib-${ZLIB_VERSION}
./configure --prefix=$PREFIX
make -j6
make check
make install

# HDF5
cd $TMP
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF_VERSION_PATH}/hdf5-${HDF_VERSION}/src/hdf5-${HDF_VERSION}.tar.gz
tar -xzf hdf5-${HDF_VERSION}.tar.gz
cd hdf5-${HDF_VERSION}
./configure --with-zlib=$PREFIX --prefix=$PREFIX --enable-hl --with-default-api-version=v18
make -j6
make check
make install
make check-install

# NetCDF
cd $TMP
wget https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-c-${NETCDF_VERSION}.tar.gz
tar -xzf netcdf-c-${NETCDF_VERSION}.tar.gz
cd netcdf-c-${NETCDF_VERSION}
CPPFLAGS=-I$PREFIX/include LDFLAGS=-L$PREFIX/lib ./configure --prefix=$PREFIX --disable-dap
make -j6
make check
make install

# JPG
cd $TMP
wget -O jpg.tar.gz https://github.com/LuaDist/libjpeg/archive/master.tar.gz
tar -xzf jpg.tar.gz
cd libjpeg-master
mkdir bld
cd bld
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release -DBUILD_STATIC=OFF -DBUILD_EXECUTABLES=OFF
make -j6
make install

# OpenJPG
cd $TMP
wget -O openjp.tar.gz https://github.com/uclouvain/openjpeg/archive/v${OPENJPG_VERSION}.tar.gz
tar -xzf openjp.tar.gz
cd openjpeg-${OPENJPG_VERSION}
mkdir bld
cd bld
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release -DBUILD_STATIC=OFF -DBUILD_EXECUTABLES=OFF
make -j6
make install

# PNG
cd $TMP
wget -O libpng.zip https://github.com/atmoswing/large-files/raw/master/libraries/libpng-1634.zip
unzip libpng.zip
cd libpng-*
mkdir bld
cd bld
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release
make -j6
make install

# Jasper
cd $TMP
wget -O jasper.tar.gz https://github.com/mdadams/jasper/archive/version-${JASPER_VERSION}.tar.gz
tar -xzf jasper.tar.gz
cd jasper-version-${JASPER_VERSION}
mkdir bld
cd bld
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release -DJAS_ENABLE_SHARED=ON -DJAS_ENABLE_LIBJPEG=ON -DJAS_ENABLE_PROGRAMS=OFF
make -j6
make install

# ecCodes
cd $TMP
wget -O eccodes.tar.gz https://confluence.ecmwf.int/download/attachments/45757960/eccodes-${ECCODES_VERSION}-Source.tar.gz
tar -xzf eccodes.tar.gz
cd eccodes-${ECCODES_VERSION}-Source
mkdir bld
cd bld
cmake .. -DCMAKE_INSTALL_PREFIX=${PREFIX} -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DENABLE_JPG=ON -DENABLE_PYTHON=OFF -DENABLE_FORTRAN=OFF -DCMAKE_PREFIX_PATH="${PREFIX}"
make -j6
make install

# Cleanup
cd $HOME
rm -rf $TMP
