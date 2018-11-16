#!/usr/bin/env sh

export HOMEBREW_PREFIX="$HOME/.libs"
export HOMEBREW_CELLAR="$HOME/.libs"
export PATH=${HOMEBREW_PREFIX}:$PATH

brew install proj
brew install jasper
brew install netcdf

# Build libraries
chmod +x ci/travis/build-wxwidgets-osx.sh
ci/travis/build-wxwidgets-osx.sh
chmod +x ci/travis/build-gdal-osx.sh
ci/travis/build-gdal-osx.sh

# Changing permissions of Homebrew libraries
sudo chmod -R 777 /usr/local/Cellar

ls $HOME/.libs/include