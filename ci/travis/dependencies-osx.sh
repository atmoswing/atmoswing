#!/usr/bin/env sh

export HOMEBREW_NO_AUTO_UPDATE=1
brew install proj
brew install netcdf

# Build libraries
chmod +x ci/travis/build-wxwidgets-osx.sh
ci/travis/build-wxwidgets-osx.sh
chmod +x ci/travis/build-gdal-osx.sh
ci/travis/build-gdal-osx.sh

# Changing permissions of Homebrew libraries
sudo chmod -R 777 /usr/local/Cellar