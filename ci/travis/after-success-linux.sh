#!/usr/bin/env sh

export LD_LIBRARY_PATH=$HOME/.libs/lib:$LD_LIBRARY_PATH
cd /home/travis/build/atmoswing/atmoswing || exit
/usr/bin/cpack -C release -G DEB
cat /home/travis/build/atmoswing/atmoswing/_CPack_Packages/Linux/DEB/PreinstallOutput.log
ls -lha
rename "s/linux64/${DISTRO}/" *linux64.deb
export PKG_FILE=$(ls *.deb)
mkdir deploy
mv $PKG_FILE deploy/
echo "Package: $PKG_FILE"