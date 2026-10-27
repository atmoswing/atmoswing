# Base on Ubuntu 22.04

# Build dependencies (vcpkg builds the libraries from source)
sudo apt-get install -y cmake ninja-build git curl zip unzip tar \
    build-essential pkg-config autoconf automake libtool ca-certificates

# Install vcpkg (skip if already installed) and point VCPKG_ROOT to it
if [ ! -d "$HOME/vcpkg" ]; then
    git clone https://github.com/microsoft/vcpkg.git "$HOME/vcpkg"
    "$HOME/vcpkg/bootstrap-vcpkg.sh" -disableMetrics
fi
export VCPKG_ROOT="$HOME/vcpkg"

git clone https://github.com/atmoswing/atmoswing.git

cd atmoswing

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_FORECASTER=OFF -DBUILD_VIEWER=OFF \
    -DBUILD_OPTIMIZER=ON -DBUILD_DOWNSCALER=OFF \
    -DUSE_GUI=OFF -DBUILD_TESTS=ON

cmake --build build
