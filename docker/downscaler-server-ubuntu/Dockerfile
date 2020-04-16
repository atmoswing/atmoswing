##
# atmoswing/downscaler-server:ubuntu

# Steps ordered them from the less frequently changed to the more frequently
# changed to ensure the build cache is reused.

FROM ubuntu:18.04 AS builder

WORKDIR /app
RUN mkdir libs
ARG LIB_DIR=/app/libs

# Setup build environment
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    gnome-devel \
    make \
    unzip \
    wget

# Install new CMake version
ARG CMAKE_VERSION=3.17.0
RUN wget --no-check-certificate --tries=20 \
        https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh \
    && mkdir /opt/cmake \
    && sh cmake-${CMAKE_VERSION}-Linux-x86_64.sh --skip-license --prefix=/opt/cmake \
    && rm -f sh cmake-${CMAKE_VERSION}-Linux-x86_64.sh \
    && ln -s /opt/cmake/bin/cmake /usr/bin/cmake \
    && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake

# Install dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    libcurl4 \
    libnetcdf-dev \
    libopenjp2-7-dev \
    libsqlite3-dev \
    libssl-dev \
    libtiff5-dev \
    libwxbase3.0-dev \
    sqlite3

# Build Proj
ARG PROJ_VERSION=7.0.0
RUN wget --no-check-certificate --tries=20 -O proj.tar.gz --no-check-certificate \
        https://download.osgeo.org/proj/proj-${PROJ_VERSION}.tar.gz \
    && tar -xzf proj.tar.gz \
    && rm -f proj.tar.gz \
    && cd proj-${PROJ_VERSION} \
    && ./configure --prefix=${LIB_DIR} \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf proj-${PROJ_VERSION}

# Build ecCodes
ARG ECCODES_VERSION=2.17.0
RUN wget --no-check-certificate --tries=20 -O eccodes.tar.gz \
        https://confluence.ecmwf.int/download/attachments/45757960/eccodes-${ECCODES_VERSION}-Source.tar.gz \
    && tar -xzf eccodes.tar.gz \
    && rm -f eccodes.tar.gz \
    && cd eccodes-${ECCODES_VERSION}-Source \
    && mkdir bld \
    && cd bld \
    && cmake .. -DCMAKE_INSTALL_PREFIX=${LIB_DIR} -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DENABLE_JPG=ON -DENABLE_PYTHON=OFF -DENABLE_FORTRAN=OFF \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && cd .. \
    && rm -rf eccodes-${ECCODES_VERSION}-Source

# Copy source code
RUN mkdir src
COPY ./ src

# Build AtmoSwing
RUN cd src \
    && cmake CMakeLists.txt -DBUILD_OPTIMIZER=0 -DBUILD_FORECASTER=0 -DBUILD_DOWNSCALER=1 -DBUILD_VIEWER=0 \
        -DON_DOCKER=ON -DCREATE_INSTALLER=OFF -DUSE_GUI=0 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$LIB_DIR \
    && make -j$(nproc)


# Build test image
FROM ubuntu:18.04 AS test

WORKDIR /app

# Install dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates \
    libcurl4 \
    libnetcdf13 \
    libopenjp2-7 \
    libtiff5 \
    libwxbase3.0-0v5 \
    openssl \
    sqlite3 \
    unzip

# Copy from builder
COPY --from=builder  /app/libs/lib/*.so /usr/local/lib/
COPY --from=builder  /app/libs/lib/*.so.* /usr/local/lib/
COPY --from=builder  /app/libs/share /usr/local/share
COPY --from=builder  /app/src/data /usr/local/share/atmoswing
COPY --from=builder  /app/src/tests/atmoswing-tests /app/
COPY --from=builder  /app/src/tests/files /app/tests/files

ENV ECCODES_DEFINITION_PATH /usr/local/share/eccodes/definitions
RUN ldconfig

RUN ./atmoswing-tests


# Build final image
FROM ubuntu:18.04 AS runner

WORKDIR /app

# Install dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates \
    libcurl4 \
    libnetcdf13 \
    libopenjp2-7 \
    libtiff5 \
    libwxbase3.0-0v5 \
    openssl \
    sqlite3 \
    unzip

# Copy from builder
COPY --from=builder  /app/libs/lib/*.so /usr/local/lib/
COPY --from=builder  /app/libs/lib/*.so.* /usr/local/lib/
COPY --from=builder  /app/libs/share /usr/local/share
COPY --from=builder  /app/src/data /usr/local/share/atmoswing
COPY --from=builder  /app/src/Release/* /usr/local/bin/

ENV ECCODES_DEFINITION_PATH /usr/local/share/eccodes/definitions
RUN ldconfig

ENTRYPOINT ["/usr/local/bin/atmoswing-downscaler"]
CMD ["--help"]
