#!/usr/bin/env sh

# Build libraries
chmod +x ci/travis/build-proj.sh
ci/travis/build-proj.sh
chmod +x ci/travis/build-eccodes.sh
ci/travis/build-eccodes.sh