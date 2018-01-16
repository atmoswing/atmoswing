# Install a recent CMake
Write-Host "`nInstalling CMake" -ForegroundColor Yellow
cd $TMP_DIR
if ($ON_APPVEYOR) {
  appveyor DownloadFile $CMAKE_URL -FileName cmake.zip > $null
} else {
  Invoke-WebRequest -Uri $CMAKE_URL -OutFile cmake.zip
}
7z x cmake.zip -o"$TMP_DIR" > $null
move "$TMP_DIR\cmake-*" "$CMAKE_DIR"
$path = $env:Path
$path = ($path.Split(';') | Where-Object { $_ -ne 'C:\Program Files (x86)\CMake\bin' }) -join ';'
$path = ($path.Split(';') | Where-Object { $_ -ne 'C:\Tools\NuGet' }) -join ';'
$env:Path = $path
$env:Path += ";$CMAKE_DIR\bin"
cmake --version

# Install wxWidgets
if(!(Test-Path -Path "$LIB_DIR\include\wx") -Or $REBUILD_WX) {
  Init-Build "wxwidgets"
  Download-Lib "wxwidgets" $WX_URL
  7z x wxwidgets.zip -o"$TMP_DIR\wxwidgets" > $null
  cd "$TMP_DIR\wxwidgets\build\msw"
  nmake -f makefile.vc BUILD=release MONOLITHIC=0 SHARED=0 USE_OPENGL=0 TARGET_CPU=$WX_TARGET_CPU > $null
  nmake -f makefile.vc BUILD=debug MONOLITHIC=0 SHARED=0 USE_OPENGL=0 TARGET_CPU=$WX_TARGET_CPU > $null
  move "$TMP_DIR\wxwidgets\include" "$LIB_DIR\include"
  copy "$TMP_DIR\wxwidgets\lib\vc_${TARGET_CPU}_lib\mswu\wx\setup.h" "$LIB_DIR\wxwidgets\include\wx\setup.h"
  move "$LIB_DIR\include\wx\msw\rcdefs.h" "$LIB_DIR\include\wx\msw\rcdefs.h_old"
  copy "$TMP_DIR\wxwidgets\lib\vc_${TARGET_CPU}_lib\mswu\wx\msw\rcdefs.h" "$LIB_DIR\include\wx\msw\rcdefs.h"
  move "$TMP_DIR\wxwidgets\lib" "$LIB_DIR\lib"
}
$env:WXWIN = "$LIB_DIR"


Invoke-Expression -Command $PSScriptRoot\libs-common-install.ps1


# Install Gdal
if(!(Test-Path -Path "$LIB_DIR\include\gdal.h") -Or $REBUILD_GDAL) {
  Init-Build "gdal"
  Download-Lib "gdal" $GDAL_URL
  7z x gdal.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\gdal-*" "$TMP_DIR\gdal"
  cd "$TMP_DIR\gdal"
  $LIB_DIR_REV=$LIB_DIR -replace '\\','/'
  nmake -f makefile.vc MSVC_VER=$MSC_VER WIN64=$GDAL_WIN64 GDAL_HOME="$LIB_DIR" CURL_DIR="$LIB_DIR" CURL_INC="-I$LIB_DIR_REV/include" CURL_LIB="$LIB_DIR_REV/lib/libcurl.lib wsock32.lib wldap32.lib winmm.lib" CURL_CFLAGS=-DCURL_STATICLIB
  nmake -f makefile.vc MSVC_VER=$MSC_VER WIN64=$GDAL_WIN64 GDAL_HOME="$LIB_DIR" CURL_DIR="$LIB_DIR" CURL_INC="-I$LIB_DIR_REV/include" CURL_LIB="$LIB_DIR_REV/lib/libcurl.lib wsock32.lib wldap32.lib winmm.lib" CURL_CFLAGS=-DCURL_STATICLIB install
  nmake -f makefile.vc MSVC_VER=$MSC_VER WIN64=$GDAL_WIN64 GDAL_HOME="$LIB_DIR" CURL_DIR="$LIB_DIR" CURL_INC="-I$LIB_DIR_REV/include" CURL_LIB="$LIB_DIR_REV/lib/libcurl.lib wsock32.lib wldap32.lib winmm.lib" CURL_CFLAGS=-DCURL_STATICLIB devinstall
}

Get-ChildItem "$LIB_DIR/include"
