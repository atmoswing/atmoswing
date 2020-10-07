#!/usr/bin/env sh

export LD_LIBRARY_PATH=$HOME/.libs/lib:$LD_LIBRARY_PATH
cd $TRAVIS_BUILD_DIR
cpack -C release -G DEB
ls -lha
rename "s/linux64/${DISTRO}/" *linux64.deb
export PKG_FILE=$(ls *.deb)
mkdir deploy
mv $PKG_FILE deploy/
echo "Package: $PKG_FILE"