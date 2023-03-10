##
# atmoswing/optimizer-server:ubuntu

# Steps ordered from the less frequently changed to the more frequently
# changed to ensure the build cache is reused.

FROM ubuntu:22.04 AS builder

WORKDIR /app
RUN mkdir libs
ARG LIB_DIR=/app/libs

# Setup build environment
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y --no-install-recommends cmake python3 python3-pip git ca-certificates build-essential

# Setup conan
RUN python3 -m pip install conan==1.*
RUN conan profile new default --detect \
    && conan remote add gitlab https://gitlab.com/api/v4/packages/conan \
    && conan profile update settings.compiler.libcxx=libstdc++11 default

# Copy source code
RUN mkdir src && mkdir src/bin
COPY ./ src

# Build AtmoSwing
WORKDIR /app/src/bin
RUN conan install .. -s build_type=Release --build=missing --build=openjpeg -o enable_tests=False -o with_gui=False  \
        -o build_forecaster=False -o build_viewer=False -o build_optimizer=True -o build_downscaler=False \
    && conan build ..

# Prepare files for deployment
RUN mkdir /app/deploy \
    && mkdir /app/deploy/bin \
    && mkdir /app/deploy/share
RUN ls /app/src/bin/ -R
RUN cp -a /app/src/data /app/deploy/share/atmoswing \
    && cp -a /app/src/bin/bin/* /app/deploy/bin/ \
    && cp -a /app/src/bin/share/* /app/deploy/share/


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

ENTRYPOINT ["/usr/bin/atmoswing-optimizer"]
CMD ["--help"]
