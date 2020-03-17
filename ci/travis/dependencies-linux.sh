#!/usr/bin/env sh

# Build libraries
chmod +x ci/travis/build-proj.sh
ci/travis/build-proj.sh
chmod +x ci/travis/build-gdal.sh
ci/travis/build-gdal.sh
chmod +x ci/travis/build-jasper.sh
ci/travis/build-jasper.sh
chmod +x ci/travis/build-eccodes.sh
ci/travis/build-eccodes.sh