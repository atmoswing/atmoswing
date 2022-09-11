$stopwatchlibs = [system.diagnostics.stopwatch]::StartNew()

# Install WIX
if ($APPVEYOR) {
  Write-Host "`nInstalling WIX" -ForegroundColor Yellow
  cd $TMP_DIR
  $WIX_URL="https://github.com/wixtoolset/wix3/releases/download/wix3111rtm/wix311-binaries.zip"
  if ($APPVEYOR) {
    appveyor DownloadFile $WIX_URL -FileName wix.zip > $null
  } else {
    Invoke-WebRequest -Uri $WIX_URL -OutFile wix.zip
  }
  7z x wix.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\wix311-binaries\*" "$WIX_DIR"
  $env:Path += ";$WIX_DIR\bin"
}

# Install wxWidgets
if(!(Test-Path -Path "$LIB_DIR\include\wx") -Or $REBUILD_WX) {
  Init-Build "wxwidgets"
  Download-Lib "wxwidgets" $WX_URL
  7z x wxwidgets.zip -o"$TMP_DIR\wxwidgets" > $null
  cd "$TMP_DIR\wxwidgets\build\msw"
  nmake -f makefile.vc BUILD=release MONOLITHIC=0 SHARED=0 USE_OPENGL=0 TARGET_CPU=$WX_TARGET_CPU > $null
  nmake -f makefile.vc BUILD=debug MONOLITHIC=0 SHARED=0 USE_OPENGL=0 TARGET_CPU=$WX_TARGET_CPU > $null
  Copy-Item "$TMP_DIR\wxwidgets\include\*" -Destination "$LIB_DIR\include" -Recurse
  copy "$TMP_DIR\wxwidgets\lib\vc_${TARGET_CPU}_lib\mswu\wx\setup.h" "$LIB_DIR\include\wx\setup.h"
  move "$LIB_DIR\include\wx\msw\rcdefs.h" "$LIB_DIR\include\wx\msw\rcdefs.h_old"
  copy "$TMP_DIR\wxwidgets\lib\vc_${TARGET_CPU}_lib\mswu\wx\msw\rcdefs.h" "$LIB_DIR\include\wx\msw\rcdefs.h"
  Copy-Item "$TMP_DIR\wxwidgets\lib\*" -Destination "$LIB_DIR\lib" -Recurse
} else {
  Write-Host "`nwxWidgets has been found in cache and will not be built" -ForegroundColor Yellow
}
$env:WXWIN = "$LIB_DIR"


. $PSScriptRoot\libs-common-install.ps1

if ($stopwatchlibs.Elapsed.TotalMinutes -gt 30) { return }


# Install PNG (for GDAL)
if(!(Test-Path -Path "$LIB_DIR\include\png.h") -Or $REBUILD_PNG) {
  Init-Build "png"
  Download-Lib "png" $PNG_URL
  7z x png.zip -o"$TMP_DIR" > $null
  cd "$TMP_DIR\libpng*"
  mkdir bld > $null
  cd bld
  cmake .. -G"$VS_VER" $CMAKE_GENERATOR -DCMAKE_INSTALL_PREFIX="$LIB_DIR" -DBUILD_STATIC=ON -DBUILD_EXECUTABLES=OFF -DCMAKE_PREFIX_PATH="$LIB_DIR" > $null
  cmake --build . --config release > $null
  cmake --build . --config release --target INSTALL > $null
} else {
  Write-Host "`nPng has been found in cache and will not be built" -ForegroundColor Yellow
}

if ($stopwatchlibs.Elapsed.TotalMinutes -gt 30) { return }

# Install Gdal
if(!(Test-Path -Path "$LIB_DIR\include\gdal.h") -Or $REBUILD_GDAL) {
  Init-Build "gdal"
  Download-Lib "gdal" $GDAL_URL
  7z x gdal.zip -o"$TMP_DIR" > $null
  cd "$TMP_DIR\gdal-*"
  $LIB_DIR_REV=$LIB_DIR -replace '\\','/'
  nmake -f makefile.vc MSVC_VER=$MSC_VER WIN64=$GDAL_WIN64 GDAL_HOME="$LIB_DIR" PROJ_INCLUDE="-I$LIB_DIR_REV/include" PROJ_LIBRARY="$LIB_DIR_REV/lib/proj.lib" CURL_DIR="$LIB_DIR" CURL_INC="-I$LIB_DIR_REV/include" CURL_LIB="$LIB_DIR_REV/lib/libcurl.lib wsock32.lib wldap32.lib winmm.lib" CURL_CFLAGS=-DCURL_STATICLIB > $null
  nmake -f makefile.vc MSVC_VER=$MSC_VER WIN64=$GDAL_WIN64 GDAL_HOME="$LIB_DIR" PROJ_INCLUDE="-I$LIB_DIR_REV/include" PROJ_LIBRARY="$LIB_DIR_REV/lib/proj.lib" CURL_DIR="$LIB_DIR" CURL_INC="-I$LIB_DIR_REV/include" CURL_LIB="$LIB_DIR_REV/lib/libcurl.lib wsock32.lib wldap32.lib winmm.lib" CURL_CFLAGS=-DCURL_STATICLIB install  > $null
  nmake -f makefile.vc MSVC_VER=$MSC_VER WIN64=$GDAL_WIN64 GDAL_HOME="$LIB_DIR" PROJ_INCLUDE="-I$LIB_DIR_REV/include" PROJ_LIBRARY="$LIB_DIR_REV/lib/proj.lib" CURL_DIR="$LIB_DIR" CURL_INC="-I$LIB_DIR_REV/include" CURL_LIB="$LIB_DIR_REV/lib/libcurl.lib wsock32.lib wldap32.lib winmm.lib" CURL_CFLAGS=-DCURL_STATICLIB devinstall > $null
} else {
  Write-Host "`nGDAL has been found in cache and will not be built" -ForegroundColor Yellow
}
