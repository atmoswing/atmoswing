#!/usr/bin/env sh

brew install proj
brew install jasper
brew install netcdf
brew upgrade openssl

brew link openssl --force
whereis openssl

# Build libraries
chmod +x ci/travis/build-wxwidgets-osx.sh
ci/travis/build-wxwidgets-osx.sh
chmod +x ci/travis/build-gdal-osx.sh
ci/travis/build-gdal-osx.sh

# Changing permissions of Homebrew libraries
sudo chmod -R 777 /usr/local/Cellar