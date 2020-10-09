#!/bin/bash

cd $TRAVIS_BUILD_DIR || exit
pwd
cpack
ls -lha
export PKG_FILE=$(ls *.dmg)
mkdir deploy
mv $PKG_FILE deploy/
echo "Package: $PKG_FILE"