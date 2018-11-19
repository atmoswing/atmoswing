#!/usr/bin/env sh

cd "$HOME/atmoswing"
cpack -C release -G DEB
ls -lha
rename "s/linux64/${DISTRO}/" *linux64.deb
export PKG_FILE=$(ls *.deb)
mkdir deploy
mv $PKG_FILE deploy/
echo "Package: $PKG_FILE"