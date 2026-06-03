##
# atmoswing/forecaster-server:ubuntu

# Steps ordered from the less frequently changed to the more frequently
# changed to ensure the build cache is reused.

FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV VCPKG_ROOT=/opt/vcpkg

# Setup build environment (vcpkg builds the dependencies from source)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake ninja-build git curl zip unzip tar ca-certificates \
        build-essential pkg-config autoconf automake libtool

# Install vcpkg
RUN git clone --depth 1 https://github.com/microsoft/vcpkg.git ${VCPKG_ROOT} \
    && ${VCPKG_ROOT}/bootstrap-vcpkg.sh -disableMetrics

# Copy source code
WORKDIR /app/src
COPY ./ .

# Build AtmoSwing (headless server: no GUI / Viewer)
RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DON_DOCKER=ON \
        -DBUILD_FORECASTER=ON \
        -DBUILD_VIEWER=OFF \
        -DBUILD_OPTIMIZER=OFF \
        -DBUILD_DOWNSCALER=OFF \
        -DUSE_GUI=OFF \
        -DBUILD_TESTS=OFF \
    && cmake --build build

# Prepare files for deployment
RUN cmake --install build --prefix /app/deploy \
    && mkdir -p /app/deploy/lib /app/deploy/share \
    && find /app/src/build -name 'libeccodes*.so*' -exec cp -a {} /app/deploy/lib/ \; \
    && cp -a /app/src/data /app/deploy/share/atmoswing \
    && cp -a /app/src/build/share/. /app/deploy/share/


# Build final image
FROM ubuntu:22.04 AS runner

WORKDIR /app
ENV ECCODES_DEFINITION_PATH=/usr/share/eccodes/definitions

# Install dependencies
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy from builder
COPY --from=builder  /app/deploy /usr
RUN ldconfig

ENTRYPOINT ["/usr/bin/atmoswing-forecaster"]
CMD ["--help"]
