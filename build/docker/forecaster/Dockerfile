##
# atmoswing/forecaster-server:ubuntu

# Steps ordered from the less frequently changed to the more frequently
# changed to ensure the build cache is reused.

FROM ubuntu:22.04 AS builder

WORKDIR /app
RUN mkdir libs
ARG LIB_DIR=/app/libs

# Setup build environment
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y cmake python3 python3-pip git ca-certificates

# Setup conan
RUN python3 -m pip install conan \
    && conan profile new default --detect \
    && conan profile update settings.compiler.libcxx=libstdc++11 default \
    && conan remote add gitlab https://gitlab.com/api/v4/packages/conan

# Copy source code
RUN mkdir src && mkdir src/bin
COPY ./ src

# Build AtmoSwing
RUN cd src/bin \
    && conan install .. -s build_type=Release --build=missing --build=openjpeg -o enable_tests=False -o with_gui=False  \
        -o build_forecaster=True -o build_viewer=False -o build_optimizer=False -o build_downscaler=False \
    && conan build ..

# Prepare files for deployment
RUN mkdir /app/deploy \
    && mkdir /app/deploy/bin \
    && mkdir /app/deploy/share \
RUN ls /app/src/bin/ -R
RUN cp -a /app/src/data /app/deploy/share/atmoswing \
    && cp -a /app/src/bin/bin/* /app/deploy/bin/ \
    && cp a /app/src/bin/share/* /app/deploy/share/


# Build final image
FROM ubuntu:22.04 AS runner

WORKDIR /app

# Install dependencies
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates

# Copy from builder
COPY --from=builder  /app/deploy /usr

ENTRYPOINT ["/usr/bin/atmoswing-forecaster"]
CMD ["--help"]
