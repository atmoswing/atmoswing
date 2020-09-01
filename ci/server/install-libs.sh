#!/usr/bin/env bash

PREFIX=${HOME}/libs
TMP=${HOME}/tmp-build

mkdir $TMP

# wxWidgets
cd $TMP
wget -q -O wxwidgets.tar.bz2 "https://github.com/wxWidgets/wxWidgets/releases/download/v3.1.2/wxWidgets-3.1.2.tar.bz2"
tar -xjf wxwidgets.tar.bz2
cd wxWidgets-3.1.2
mkdir wxbase-build
cd wxbase-build
../configure --enable-unicode --disable-monolithic --disable-shared --disable-debug --disable-gui --prefix=$PREFIX
make -j6
make install

# cURL
cd $TMP
wget https://curl.haxx.se/download/curl-7.65.1.tar.gz
tar -xzf curl-7.65.1.tar.gz
cd curl-7.65.1
./configure --prefix=$PREFIX
make -j6
make check
make install

# Proj
cd $TMP
wget https://download.osgeo.org/proj/proj-5.2.0.tar.gz
tar -xzf proj-5.2.0.tar.gz
cd proj-5.2.0
./configure --prefix=$PREFIX
make -j6
make install

# Zlib
cd $TMP
wget https://zlib.net/zlib-1.2.11.tar.gz
tar -xzf zlib-1.2.11.tar.gz
cd zlib-1.2.11
./configure --prefix=$PREFIX
make -j6
make check
make install

# HDF5
cd $TMP
wget https://s3.amazonaws.com/hdf-wordpress-1/wp-content/uploads/manual/HDF5/HDF5_1_10_5/source/hdf5-1.10.5.tar.gz
tar -xzf hdf5-1.10.5.tar.gz
cd hdf5-1.10.5
./configure --with-zlib=$PREFIX --prefix=$PREFIX --enable-hl
make -j6
make check
make install
make check-install

# NetCDF
cd $TMP
wget https://www.unidata.ucar.edu/downloads/netcdf/ftp/netcdf-c-4.7.0.tar.gz
tar -xzf netcdf-c-4.7.0.tar.gz
cd netcdf-c-4.7.0
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
wget -O openjp.tar.gz https://github.com/uclouvain/openjpeg/archive/v2.3.1.tar.gz
tar -xzf openjp.tar.gz
cd openjpeg-2.3.1
mkdir bld
cd bld
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release -DBUILD_STATIC=OFF -DBUILD_EXECUTABLES=OFF
make -j6
make install

# Jasper
cd $TMP
wget -O jasper.tar.gz https://github.com/mdadams/jasper/archive/version-2.0.16.tar.gz
tar -xzf jasper.tar.gz
cd jasper-version-2.0.16
mkdir bld
cd bld
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release -DJAS_ENABLE_SHARED=ON -DJAS_ENABLE_LIBJPEG=ON -DJAS_ENABLE_PROGRAMS=OFF
make -j6
make install

# ecCodes
cd $TMP
wget -O eccodes.tar.gz https://confluence.ecmwf.int/download/attachments/45757960/eccodes-2.12.0-Source.tar.gz
tar -xzf eccodes.tar.gz
cd eccodes-2.12.0-Source
mkdir bld
cd bld
cmake .. -DCMAKE_INSTALL_PREFIX=${PREFIX} -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DENABLE_JPG=ON -DENABLE_PYTHON=OFF -DENABLE_FORTRAN=OFF -DCMAKE_PREFIX_PATH="${PREFIX}"
make -j6
make install

# Cleanup
cd $HOME
rm -rf $TMP
