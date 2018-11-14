# Options
$MSC_VER=1915
if ($env:APPVEYOR) {
  $TMP_DIR="C:\projects\tmp"
  $LIB_DIR="C:\projects\libs"
  $CMAKE_DIR="C:\projects\cmake"
  $WIX_DIR="C:\projects\wix"
} else {
  $TMP_DIR="C:\Users\$env:UserName\Downloads\tmp"
  $LIB_DIR="C:\Users\$env:UserName\AtmoSwing-libs"
  $CMAKE_DIR="C:\Program Files\CMake\bin"
  $WIX_DIR="C:\Program Files\WiX"
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

# Libraries URL
$WX_URL="https://github.com/wxWidgets/wxWidgets/releases/download/v3.1.0/wxWidgets-3.1.0.zip"
$ZLIB_URL="http://www.zlib.net/zlib1211.zip"
$JPEG_URL="https://github.com/LuaDist/libjpeg/archive/master.zip"
$PNG_URL="https://github.com/atmoswing/large-files/raw/master/libraries/libpng-1634.zip"
$JASPER_URL="https://github.com/mdadams/jasper/archive/version-2.0.14.zip"
$CURL_URL="https://github.com/curl/curl/archive/curl-7_54_1.zip"
$PROJ_URL="https://github.com/OSGeo/proj.4/archive/4.9.3.zip"
$HDF5_URL="https://github.com/atmoswing/large-files/raw/master/libraries/CMake-hdf5-1.10.1.zip"
$NETCDF_URL="https://github.com/atmoswing/large-files/raw/master/libraries/netcdf-4.5.0.zip"
$GDAL_URL="http://download.osgeo.org/gdal/2.2.3/gdal223.zip"

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

