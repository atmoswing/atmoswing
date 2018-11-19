#!/bin/bash

cd /Users/travis/build/atmoswing/atmoswing
cpack
ls -lha
export PKG_FILE=$(ls *.dmg)
mkdir deploy
mv $PKG_FILE deploy/
echo "Package: $PKG_FILE"