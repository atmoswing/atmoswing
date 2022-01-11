# Options
if ($env:APPVEYOR) {
  $MSC_VER=1925
  $VS_VER_NB="17"
  $VS_VER_YR="2022"
  $CMAKE_GENERATOR="-Ax64"
  $TMP_DIR="C:\projects\tmp"
  $LIB_DIR="C:\projects\libs"
  $CMAKE_DIR="C:\projects\cmake"
  $WIX_DIR="C:\projects\wix"
  $PATCH_DIR="C:\projects\atmoswing\ci\appveyor\patches"
  $BASH_DIR="C:\Program Files\Git\bin"
  $BASH_PATH="C:\Program Files\Git\bin\bash.exe"
} else {
  $MSC_VER=1925
  $VS_VER_NB="17"
  $VS_VER_YR="2022"
  $CMAKE_GENERATOR="-Ax64"
  $TMP_DIR="$env:UserProfile\Downloads\tmp"
  $LIB_DIR="$env:UserProfile\AtmoSwing-libs\vs-$VS_VER_YR"
  $CMAKE_DIR="C:\Program Files\CMake\bin"
  $WIX_DIR="C:\Program Files\WiX"
  $PATCH_DIR=".\patches"
  $BASH_DIR="C:\Program Files\Git\bin"
  $BASH_PATH="C:\Program Files\Git\bin\bash.exe"
}

$VS_VER="Visual Studio $VS_VER_NB $VS_VER_YR Win64"
$PROGRAM_FILES="Program Files (x86)"
if ($VS_VER_YR -ge "2019") {
  $VS_VER="Visual Studio $VS_VER_NB $VS_VER_YR"
}
if ($VS_VER_YR -ge "2022") {
  $PROGRAM_FILES="Program Files"
}

# Force rebuilding some libraries
$REBUILD_WX=$false
$REBUILD_ZLIB=$false
$REBUILD_JPEG=$false
$REBUILD_PNG=$false
$REBUILD_TIFF=$false
$REBUILD_JASPER=$false
$REBUILD_CURL=$false
$REBUILD_PROJ=$false
$REBUILD_HDF5=$false
$REBUILD_NETCDF=$false
$REBUILD_GDAL=$false
$REBUILD_ECCODES=$false
$REBUILD_SQLITE=$false

# Libraries URL
$WX_URL="https://github.com/wxWidgets/wxWidgets/releases/download/v3.1.5/wxWidgets-3.1.5.zip"
$ZLIB_URL="http://www.zlib.net/zlib1211.zip"
$JPEG_URL="https://github.com/LuaDist/libjpeg/archive/master.zip"
$OPENJPEG_URL="https://github.com/uclouvain/openjpeg/archive/v2.3.1.zip"
$PNG_URL="https://github.com/atmoswing/large-files/raw/master/libraries/libpng-1634.zip"
$TIFF_URL="https://gitlab.com/libtiff/libtiff/-/archive/v4.3.0/libtiff-v4.3.0.zip"
$JASPER_URL="https://github.com/jasper-software/jasper/archive/refs/tags/version-2.0.33.zip"
$CURL_URL="https://github.com/curl/curl/releases/download/curl-7_80_0/curl-7.80.0.zip"
$PROJ_URL="https://github.com/OSGeo/PROJ/releases/download/8.2.0/proj-8.2.0.zip"
$HDF5_URL="https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.13/hdf5-1.13.0/src/CMake-hdf5-1.13.0.zip"
$NETCDF_URL="https://github.com/Unidata/netcdf-c/archive/refs/tags/v4.8.1.zip"
$GDAL_URL="https://github.com/OSGeo/gdal/releases/download/v3.4.0/gdal340.zip"
$ECCODES_URL="https://github.com/ecmwf/eccodes/archive/refs/tags/2.24.1.tar.gz"
$SQLITE_SRC_URL="https://www.sqlite.org/2021/sqlite-amalgamation-3370000.zip"
$SQLITE_DLL_URL="https://www.sqlite.org/2021/sqlite-dll-win64-x64-3370000.zip"
$SQLITE_TOOLS_URL="https://www.sqlite.org/2021/sqlite-tools-win32-x86-3370000.zip"

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
if(!(Test-Path -Path $LIB_DIR/bin)) {
  mkdir $LIB_DIR/bin > $null
}
if(!(Test-Path -Path $LIB_DIR/include)) {
  mkdir $LIB_DIR/include > $null
}
if(!(Test-Path -Path $LIB_DIR/lib)) {
  mkdir $LIB_DIR/lib > $null
}
if(!(Test-Path -Path $TMP_DIR)) {
  mkdir $TMP_DIR > $null
}

