# Options
if ($env:APPVEYOR) {
  $MSC_VER=1923
  $VS_VER_NB="16"
  $VS_VER_YR="2019"
  $CMAKE_GENERATOR="-Ax64"
  $TMP_DIR="C:\projects\tmp"
  $LIB_DIR="C:\projects\libs"
  $CMAKE_DIR="C:\projects\cmake"
  $WIX_DIR="C:\projects\wix"
  $PATCH_DIR="C:\projects\atmoswing\ci\appveyor\patches"
} else {
  $MSC_VER=1923
  $VS_VER_NB="16"
  $VS_VER_YR="2019"
  $CMAKE_GENERATOR="-Ax64"
  $TMP_DIR="$env:UserProfile\Downloads\tmp"
  $LIB_DIR="$env:UserProfile\AtmoSwing-libs\vs-$VS_VER_YR"
  $CMAKE_DIR="C:\Program Files\CMake\bin"
  $WIX_DIR="C:\Program Files\WiX"
  $PATCH_DIR=".\patches"
}

$VS_VER="Visual Studio $VS_VER_NB $VS_VER_YR Win64"
if ($VS_VER_YR -ge "2019") {
  $VS_VER="Visual Studio $VS_VER_NB $VS_VER_YR"
}

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
$REBUILD_ECCODES=$false
$REBUILD_SQLITE=$false

# Libraries URL
$WX_URL="https://github.com/wxWidgets/wxWidgets/releases/download/v3.1.3/wxWidgets-3.1.3.zip"
$ZLIB_URL="http://www.zlib.net/zlib1211.zip"
$JPEG_URL="https://github.com/LuaDist/libjpeg/archive/master.zip"
$PNG_URL="https://github.com/atmoswing/large-files/raw/master/libraries/libpng-1634.zip"
$JASPER_URL="https://github.com/mdadams/jasper/archive/version-2.0.16.zip"
$CURL_URL="https://github.com/curl/curl/archive/curl-7_64_1.zip"
$PROJ_URL="https://github.com/OSGeo/PROJ/releases/download/6.2.1/proj-6.2.1.zip"
$HDF5_URL="https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/CMake-hdf5-1.10.5.zip"
$NETCDF_URL="https://github.com/Unidata/netcdf-c/archive/v4.7.3.zip"
$GDAL_URL="https://github.com/OSGeo/gdal/releases/download/v3.0.2/gdal302.zip"
$ECCODES_URL="https://confluence.ecmwf.int/download/attachments/45757960/eccodes-2.14.1-Source.tar.gz"
$SQLITE_SRC_URL="https://www.sqlite.org/2019/sqlite-amalgamation-3270200.zip"
$SQLITE_DLL_URL="https://www.sqlite.org/2019/sqlite-dll-win64-x64-3270200.zip"
$SQLITE_TOOLS_URL="https://www.sqlite.org/2019/sqlite-tools-win32-x86-3270200.zip"

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

