##
# atmoswing/atmoswing:ubuntu-server

FROM ubuntu:18.04 AS builder

ARG CMAKE_VERSION=3.17.0-rc3
ARG PROJ_VERSION=7.0.0
ARG ECCODES_VERSION=2.17.0

MAINTAINER Pascal Horton <pascal.horton@alumnil.unil.ch>

# Setup build environment
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends git build-essential \
        gnome-devel make wget unzip

# Install new CMake version
RUN wget --no-check-certificate https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh \
    && mkdir /opt/cmake \
    && sh cmake-3.17.0-rc3-Linux-x86_64.sh --skip-license --prefix=/opt/cmake \
    && ln -s /opt/cmake/bin/cmake /usr/bin/cmake \
    && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

# Install dependencies
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends libnetcdf-dev libcurl4 libwxbase3.0-dev \
        sqlite3 libsqlite3-dev libopenjp2-7-dev libssl-dev libtiff5-dev

# Build Proj
RUN wget --no-check-certificate -O proj.tar.gz --no-check-certificate https://download.osgeo.org/proj/proj-${PROJ_VERSION}.tar.gz \
    && tar -xzf proj.tar.gz \
    && rm -f proj.tar.gz \
    && cd proj-${PROJ_VERSION} \
    && ./configure \
    && make -j6 \
    && make install \
    && cd .. \
    && rm -rf proj-${PROJ_VERSION}

# Build ecCodes
RUN wget --no-check-certificate -O eccodes.tar.gz https://confluence.ecmwf.int/download/attachments/45757960/eccodes-${ECCODES_VERSION}-Source.tar.gz \
    && tar -xzf eccodes.tar.gz \
    && rm -f eccodes.tar.gz \
    && cd eccodes-${ECCODES_VERSION}-Source \
    && mkdir bld \
    && cd bld \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DENABLE_JPG=ON -DENABLE_PYTHON=OFF -DENABLE_FORTRAN=OFF \
    && make -j6 \
    && make install \
    && cd .. \
    && cd .. \
    && rm -rf eccodes-${ECCODES_VERSION}-Source

# Build AtmoSwing
RUN cmake CMakeLists.txt -DBUILD_OPTIMIZER=1 -DBUILD_FORECASTER=1 -DBUILD_DOWNSCALER=1 -DBUILD_VIEWER=0 \
        -DCREATE_INSTALLER=OFF -DUSE_GUI=0 -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc)

# Test
RUN cd tests \
    && ./atmoswing-tests





# Build final image
FROM ubuntu:18.04 AS runner

# Install dependencies
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends sqlite3 libtiff5 libcurl4 \
        curl unzip ca-certificates

RUN ldconfig