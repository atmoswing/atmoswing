# Options
$TMP_DIR="C:\projects\tmp"
$LIB_DIR="C:\projects\libs"
$CMAKE_DIR="C:\projects\cmake"
$MSC_VER=1911
$VS_VER="Visual Studio 15 2017 Win64"
$ON_APPVEYOR=$true

# Force rebuilding some libraries
$REBUILD_WX=$false
$REBUILD_ZLIB=$false
$REBUILD_JPEG=$false
$REBUILD_PNG=$false
$REBUILD_JASPER=$false
$REBUILD_CURL=$false
$REBUILD_PROJ=$false
$REBUILD_HDF5=$false
$REBUILD_NETCDF=$false
$REBUILD_GDAL=$false

# Libraries URL
$CMAKE_URL="https://cmake.org/files/v3.10/cmake-3.10.0-win64-x64.zip"
$WX_URL="https://github.com/wxWidgets/wxWidgets/releases/download/v3.1.0/wxWidgets-3.1.0.zip"
$ZLIB_URL="http://www.zlib.net/zlib1211.zip"
$JPEG_URL="https://github.com/LuaDist/libjpeg/archive/master.zip"
$PNG_URL="https://github.com/atmoswing/large-files/raw/master/libraries/libpng-1634.zip"
$JASPER_URL="https://github.com/mdadams/jasper/archive/version-2.0.14.zip"
$CURL_URL="https://github.com/curl/curl/archive/curl-7_54_1.zip"
$PROJ_URL="https://github.com/OSGeo/proj.4/archive/4.9.3.zip"
$HDF5_URL="https://github.com/atmoswing/large-files/raw/master/libraries/CMake-hdf5-1.10.1.zip"
$NETCDF_URL="ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4.5.0.zip"
$GDAL_URL="http://download.osgeo.org/gdal/2.2.3/gdal223.zip"

# Setup VS environment
# https://stackoverflow.com/questions/2124753/how-can-i-use-powershell-with-the-visual-studio-command-prompt
pushd 'C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build'    
cmd /c "vcvars64.bat&set" |
foreach {
  if ($_ -match "=") {
    $v = $_.split("="); set-item -force -path "ENV:\$($v[0])"  -value "$($v[1])"
  }
}
popd
Write-Host "`nVisual Studio 2017 Command Prompt variables set." -ForegroundColor Yellow

# Define some functions
function Init-Build($name)
{
	Write-Host "`nBuilding $name" -ForegroundColor Yellow
	cd $TMP_DIR
}
function Download-Lib($name, $url)
{
	Write-Host "`nDownloading $name from $url" -ForegroundColor Yellow
	Invoke-WebRequest -Uri $url -OutFile "$name.zip"
}

# All external dependencies are installed in the defined directory
if(!(Test-Path -Path $LIB_DIR)) {
  mkdir $LIB_DIR > $null
}
if(!(Test-Path -Path $TMP_DIR)) {
  mkdir $TMP_DIR > $null
}

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
  nmake -f makefile.vc BUILD=release MONOLITHIC=0 SHARED=0 USE_OPENGL=0 TARGET_CPU=AMD64 > $null
  nmake -f makefile.vc BUILD=debug MONOLITHIC=0 SHARED=0 USE_OPENGL=0 TARGET_CPU=AMD64 > $null
  move "$TMP_DIR\wxwidgets\include" "$LIB_DIR\include"
  copy "$TMP_DIR\wxwidgets\lib\vc_x64_lib\mswu\wx\setup.h" "$LIB_DIR\wxwidgets\include\wx\setup.h"
  move "$LIB_DIR\include\wx\msw\rcdefs.h" "$LIB_DIR\include\wx\msw\rcdefs.h_old"
  copy "$TMP_DIR\wxwidgets\lib\vc_x64_lib\mswu\wx\msw\rcdefs.h" "$LIB_DIR\include\wx\msw\rcdefs.h"
  move "$TMP_DIR\wxwidgets\lib" "$LIB_DIR\lib"
}
$env:WXWIN = "$LIB_DIR"

# Install Zlib
if(!(Test-Path -Path "$LIB_DIR\include\zlib.h") -Or $REBUILD_ZLIB) {
  Init-Build "zlib"
  Download-Lib "zlib" $ZLIB_URL
  7z x zlib.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\zlib-*" "$TMP_DIR\zlib"
  cd "$TMP_DIR\zlib"
  mkdir bld > $null
  cd bld
  cmake .. -G"$VS_VER" -DCMAKE_INSTALL_PREFIX="$LIB_DIR" > $null
  cmake --build . --config release > $null
  cmake --build . --config release --target INSTALL > $null
}

# Install Jpeg
if(!(Test-Path -Path "$LIB_DIR\include\jpeglib.h") -Or $REBUILD_JPEG) {
  Init-Build "jpeg"
  Download-Lib "jpeg" $JPEG_URL
  7z x jpeg.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\libjpeg-*" "$TMP_DIR\jpeg"
  cd "$TMP_DIR\jpeg"
  mkdir bld > $null
  cd bld
  cmake .. -G"$VS_VER" -DCMAKE_INSTALL_PREFIX="$LIB_DIR" -DBUILD_STATIC=ON -DBUILD_EXECUTABLES=OFF > $null
  cmake --build . --config release > $null
  cmake --build . --config release --target INSTALL > $null
}

# Install PNG
if(!(Test-Path -Path "$LIB_DIR\include\png.h") -Or $REBUILD_PNG) {
  Init-Build "png"
  Download-Lib "png" $PNG_URL
  7z x png.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\libpng*" "$TMP_DIR\png"
  cd "$TMP_DIR\png"
  mkdir bld > $null
  cd bld
  cmake .. -G"$VS_VER" -DCMAKE_INSTALL_PREFIX="$LIB_DIR" -DBUILD_STATIC=ON -DBUILD_EXECUTABLES=OFF -DCMAKE_PREFIX_PATH="$LIB_DIR" > $null
  cmake --build . --config release > $null
  cmake --build . --config release --target INSTALL > $null
}

# Install Jasper
if(!(Test-Path -Path "$LIB_DIR\include\jasper") -Or $REBUILD_JASPER) {
  Init-Build "jasper"
  Download-Lib "jasper" $JASPER_URL
  7z x jasper.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\jasper-*" "$TMP_DIR\jasper"
  cd "$TMP_DIR\jasper"
  mkdir bld > $null
  cd bld
  cmake .. -G"$VS_VER" -DCMAKE_INSTALL_PREFIX="$LIB_DIR" -DCMAKE_BUILD_TYPE=Release -DJAS_ENABLE_SHARED=OFF -DJAS_ENABLE_LIBJPEG=ON -DJAS_ENABLE_PROGRAMS=OFF -DCMAKE_INCLUDE_PATH="$LIB_DIR\include" -DCMAKE_LIBRARY_PATH="$LIB_DIR\lib" > $null
  cmake --build . --config release > $null
  cmake --build . --config release --target INSTALL > $null
}

# Install curl
if(!(Test-Path -Path "$LIB_DIR\include\curl") -Or $REBUILD_CURL) {
  Init-Build "curl"
  Download-Lib "curl" $CURL_URL
  7z x curl.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\curl-*" "$TMP_DIR\curl"
  cd "$TMP_DIR\curl\winbuild"
  nmake -f Makefile.vc mode=dll VC=14 DEBUG=NO MACHINE=x64 > $null
  Copy-Item "$TMP_DIR\curl\builds\libcurl-vc14-x64-release-dll-ipv6-sspi-winssl\bin\*" "$LIB_DIR\bin"
  Copy-Item "$TMP_DIR\curl\builds\libcurl-vc14-x64-release-dll-ipv6-sspi-winssl\include\*" "$LIB_DIR\include"
  Copy-Item "$TMP_DIR\curl\builds\libcurl-vc14-x64-release-dll-ipv6-sspi-winssl\lib\*" "$LIB_DIR\lib"
}

# Install Proj
if(!(Test-Path -Path "$LIB_DIR\include\proj_api.h") -Or $REBUILD_PROJ) {
  Init-Build "proj"
  Download-Lib "proj" $PROJ_URL
  7z x proj.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\proj.4-*" "$TMP_DIR\proj"
  cd "$TMP_DIR\proj"
  nmake -f makefile.vc INSTDIR="$LIB_DIR" > $null
  nmake -f makefile.vc INSTDIR="$LIB_DIR" install-all > $null
}

# Install HDF5
if(!(Test-Path -Path "$LIB_DIR\include\hdf5.h") -Or $REBUILD_HDF5) {
  Init-Build "hdf5"
  Download-Lib "hdf5" $HDF5_URL
  7z x hdf5.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\CMake-hdf5-*" "$TMP_DIR\hdf5"
  cd "$TMP_DIR\hdf5"
  move "hdf5-*" "hdf5"
  cd "hdf5"
  mkdir bld > $null
  cd bld
  cmake .. -G"$VS_VER" -DCMAKE_INSTALL_PREFIX="$LIB_DIR" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF -DHDF5_BUILD_TOOLS=OFF -DHDF5_ENABLE_Z_LIB_SUPPORT=ON -DCMAKE_PREFIX_PATH="$LIB_DIR" > $null
  cmake --build . --config release > $null
  cmake --build . --config release --target INSTALL > $null
}

# Install NetCDF
if(!(Test-Path -Path "$LIB_DIR\include\netcdf.h") -Or $REBUILD_NETCDF) {
  Init-Build "netcdf"
  Download-Lib "netcdf" $NETCDF_URL
  7z x netcdf.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\netcdf-*" "$TMP_DIR\netcdf"
  cd "$TMP_DIR\netcdf"
  mkdir bld > $null
  cd bld
  $LIB_DIR_REV=$LIB_DIR -replace '\\','/'
  cmake .. -G"$VS_VER" -DCMAKE_INSTALL_PREFIX="$LIB_DIR_REV" -DCMAKE_BUILD_TYPE=Release -DENABLE_NETCDF_4=ON -DENABLE_DAP=OFF -DUSE_DAP=OFF -DHDF5_DIR="$LIB_DIR_REV/cmake" -DHDF5_C_LIBRARY="$LIB_DIR_REV/lib/libhdf5.lib" -DHDF5_HL_LIBRARY="$LIB_DIR_REV/lib/libhdf5_hl.lib" -DHDF5_INCLUDE_DIR="$LIB_DIR_REV/include" -DZLIB_INCLUDE_DIR="$LIB_DIR_REV/include" -DZLIB_LIBRARY="$LIB_DIR_REV/lib/zlib.lib" -DCMAKE_INCLUDE_PATH="$LIB_DIR_REV/include" > $null
  cmake --build . --config release > $null
  cmake --build . --config release --target INSTALL > $null
}

# Install Gdal
if(!(Test-Path -Path "$LIB_DIR\include\gdal.h") -Or $REBUILD_GDAL) {
  Init-Build "gdal"
  Download-Lib "gdal" $GDAL_URL
  7z x gdal.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\gdal-*" "$TMP_DIR\gdal"
  cd "$TMP_DIR\gdal"
  $LIB_DIR_REV=$LIB_DIR -replace '\\','/'
  nmake -f makefile.vc MSVC_VER=$MSC_VER WIN64=1 GDAL_HOME="$LIB_DIR" CURL_DIR="$LIB_DIR" CURL_INC="-I$LIB_DIR_REV/include" CURL_LIB="$LIB_DIR_REV/lib/libcurl.lib wsock32.lib wldap32.lib winmm.lib" CURL_CFLAGS=-DCURL_STATICLIB
  nmake -f makefile.vc MSVC_VER=$MSC_VER WIN64=1 GDAL_HOME="$LIB_DIR" CURL_DIR="$LIB_DIR" CURL_INC="-I$LIB_DIR_REV/include" CURL_LIB="$LIB_DIR_REV/lib/libcurl.lib wsock32.lib wldap32.lib winmm.lib" CURL_CFLAGS=-DCURL_STATICLIB install
  nmake -f makefile.vc MSVC_VER=$MSC_VER WIN64=1 GDAL_HOME="$LIB_DIR" CURL_DIR="$LIB_DIR" CURL_INC="-I$LIB_DIR_REV/include" CURL_LIB="$LIB_DIR_REV/lib/libcurl.lib wsock32.lib wldap32.lib winmm.lib" CURL_CFLAGS=-DCURL_STATICLIB devinstall
}

Get-ChildItem "$LIB_DIR/include"
