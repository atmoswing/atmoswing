#!/bin/bash

REBUILD_WX=false
WX_VERSION=3.1.4

# Build wxWidgets
if [ ! "$(ls -A ${HOME}/.libs/include/wx-3.1)" ] || [ "$REBUILD_WX" = true ]; then
  wget -q -O wxwidgets.tar.bz2 "https://github.com/wxWidgets/wxWidgets/releases/download/v${WX_VERSION}/wxWidgets-${WX_VERSION}.tar.bz2" > /dev/null
  tar -xjf wxwidgets.tar.bz2
  cd wxWidgets-${WX_VERSION}
  export LDFLAGS="-stdlib=libc++"
  export OBJCXXFLAGS="-stdlib=libc++ -std=c++11"
  ./configure --prefix=${HOME}/.libs --enable-unicode --disable-shared --enable-mediactrl=no --silent --with-macosx-version-min=10.10
  make -j4 > /dev/null
  make install
  cd ..
  printf 'wxWidgets has been built.\n'
else 
  printf 'wxWidgets will not be built (%s/.libs/include/wx-3.1 found).\n' "$HOME"
fi
