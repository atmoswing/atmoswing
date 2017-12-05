# Options
$TMP_DIR="C:\projects\tmp"
$LIB_DIR="C:\projects\libs"
$CMAKE_DIR="C:\projects\cmake"
$MSC_VER=1911
$ON_APPVEYOR=$true
$WITH_DEBUG_LIBS=$false

# Force rebuilding some libraries
$REBUILD_WX=$false
$REBUILD_CURL=$false
$REBUILD_PROJ=$false
$REBUILD_ZLIB=$false
$REBUILD_HDF5=$false
$REBUILD_NETCDF=$false
$REBUILD_GDAL=$false

# Libraries URL
$CMAKE_URL="https://cmake.org/files/v3.10/cmake-3.10.0-win64-x64.zip"
$WX_URL="https://github.com/wxWidgets/wxWidgets/releases/download/v3.1.0/wxWidgets-3.1.0.zip"
$CURL_URL="https://github.com/curl/curl/archive/curl-7_54_1.zip"
$PROJ_URL="https://github.com/OSGeo/proj.4/archive/4.9.3.zip"
$ZLIB_URL="http://www.zlib.net/zlib1211.zip"
$HDF5_URL="http://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.1/src/CMake-hdf5-1.10.1.zip"
$NETCDF_URL="ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4.5.0.zip"
$GDAL_URL="http://download.osgeo.org/gdal/2.2.3/gdal223.zip"
$JASPER_URL="http://www.nco.ncep.noaa.gov/pmb/codes/GRIB2/jasper-1.900.1.zip"


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
if(!(Test-Path -Path "$LIB_DIR\wxwidgets") -Or $REBUILD_WX) {
  Write-Host "`nBuilding wxWidgets" -ForegroundColor Yellow
  cd $TMP_DIR
  if(Test-Path -Path "$LIB_DIR\wxwidgets") {
    Remove-Item "$LIB_DIR\wxwidgets" -Force -Recurse
  }
  mkdir "$LIB_DIR\wxwidgets" > $null
  if ($ON_APPVEYOR) {
    appveyor DownloadFile $WX_URL -FileName wxwidgets.zip > $null
  } else {
    Invoke-WebRequest -Uri $WX_URL -OutFile wxwidgets.zip
  }
  7z x wxwidgets.zip -o"$TMP_DIR\wxwidgets" > $null
  cd "$TMP_DIR\wxwidgets\build\msw"
  nmake -f makefile.vc BUILD=release MONOLITHIC=0 SHARED=0 USE_OPENGL=0 TARGET_CPU=AMD64 > $null
  nmake -f makefile.vc BUILD=debug MONOLITHIC=0 SHARED=0 USE_OPENGL=0 TARGET_CPU=AMD64 > $null
  move "$TMP_DIR\wxwidgets\include" "$LIB_DIR\wxwidgets\include"
  copy "$TMP_DIR\wxwidgets\lib\vc_x64_lib\mswu\wx\setup.h" "$LIB_DIR\wxwidgets\include\wx\setup.h"
  move "$LIB_DIR\wxwidgets\include\wx\msw\rcdefs.h" "$LIB_DIR\wxwidgets\include\wx\msw\rcdefs.h_old"
  copy "$TMP_DIR\wxwidgets\lib\vc_x64_lib\mswu\wx\msw\rcdefs.h" "$LIB_DIR\wxwidgets\include\wx\msw\rcdefs.h"
  move "$TMP_DIR\wxwidgets\lib" "$LIB_DIR\wxwidgets\lib"
}
$env:WXWIN = "$LIB_DIR\wxwidgets"

# Install curl
if(!(Test-Path -Path "$LIB_DIR\curl") -Or $REBUILD_CURL) {
  Write-Host "`nBuilding curl" -ForegroundColor Yellow
  cd $TMP_DIR
  if(Test-Path -Path "$LIB_DIR\curl") {
    Remove-Item "$LIB_DIR\curl" -Force -Recurse
  }
  mkdir "$LIB_DIR\curl" > $null
  if ($ON_APPVEYOR) {
    appveyor DownloadFile $CURL_URL -FileName curl.zip > $null
  } else {
    Invoke-WebRequest -Uri $CURL_URL -OutFile curl.zip
  }
  7z x curl.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\curl-*" "$TMP_DIR\curl"
  cd "$TMP_DIR\curl\winbuild"
  nmake -f Makefile.vc mode=dll VC=14 DEBUG=NO MACHINE=x64 > $null
  move "$TMP_DIR\curl\builds\libcurl-vc14-x64-release-dll-ipv6-sspi-winssl\bin" "$LIB_DIR\curl\bin"
  move "$TMP_DIR\curl\builds\libcurl-vc14-x64-release-dll-ipv6-sspi-winssl\include" "$LIB_DIR\curl\include"
  move "$TMP_DIR\curl\builds\libcurl-vc14-x64-release-dll-ipv6-sspi-winssl\lib" "$LIB_DIR\curl\lib"
}

# Install Proj
if(!(Test-Path -Path "$LIB_DIR\proj") -Or $REBUILD_PROJ) {
  Write-Host "`nBuilding Proj" -ForegroundColor Yellow
  cd $TMP_DIR
  if(Test-Path -Path "$LIB_DIR\proj") {
    Remove-Item "$LIB_DIR\proj" -Force -Recurse
  }
  mkdir "$LIB_DIR\proj" > $null
  if ($ON_APPVEYOR) {
    appveyor DownloadFile $PROJ_URL -FileName proj.zip > $null
  } else {
    Invoke-WebRequest -Uri $PROJ_URL -OutFile proj.zip
  }
  7z x proj.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\proj.4-*" "$TMP_DIR\proj"
  cd "$TMP_DIR\proj"
  nmake -f makefile.vc INSTDIR="$LIB_DIR\proj" > $null
  nmake -f makefile.vc INSTDIR="$LIB_DIR\proj" install-all > $null
}

# Install Zlib
if(!(Test-Path -Path "$LIB_DIR\zlib") -Or $REBUILD_ZLIB) {
  Write-Host "`nBuilding Zlib" -ForegroundColor Yellow
  cd $TMP_DIR
  if(Test-Path -Path "$LIB_DIR\zlib") {
    Remove-Item "$LIB_DIR\zlib" -Force -Recurse
  }
  mkdir "$LIB_DIR\zlib" > $null
  if ($ON_APPVEYOR) {
    appveyor DownloadFile $ZLIB_URL -FileName zlib.zip > $null
  } else {
    Invoke-WebRequest -Uri $ZLIB_URL -OutFile zlib.zip
  }
  7z x zlib.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\zlib-*" "$TMP_DIR\zlib"
  cd "$TMP_DIR\zlib"
  mkdir bld > $null
  cd bld
  cmake .. -G"Visual Studio 15 2017 Win64" -DCMAKE_INSTALL_PREFIX="$LIB_DIR\zlib" > $null
  cmake --build . --config release > $null
  cmake --build . --config release --target INSTALL > $null
}

# Install HDF5
if(!(Test-Path -Path "$LIB_DIR\hdf5") -Or $REBUILD_HDF5) {
  Write-Host "`nBuilding HDF5" -ForegroundColor Yellow
  cd $TMP_DIR
  if(Test-Path -Path "$LIB_DIR\hdf5") {
    Remove-Item "$LIB_DIR\hdf5" -Force -Recurse
  }
  mkdir "$LIB_DIR\hdf5" > $null
  if ($ON_APPVEYOR) {
    appveyor DownloadFile $HDF5_URL -FileName hdf5.zip > $null
  } else {
    Invoke-WebRequest -Uri $HDF5_URL -OutFile hdf5.zip
  }
  7z x hdf5.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\CMake-hdf5-*" "$TMP_DIR\hdf5"
  cd "$TMP_DIR\hdf5"
  move "hdf5-*" "hdf5"
  cd "hdf5"
  mkdir bld > $null
  cd bld
  cmake .. -G"Visual Studio 15 2017 Win64" -DCMAKE_INSTALL_PREFIX="$LIB_DIR\hdf5" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF -DHDF5_BUILD_TOOLS=OFF -DHDF5_ENABLE_Z_LIB_SUPPORT=ON -DZLIB_LIBRARIES="$LIB_DIR/zlib/lib/zlib.lib" -DZLIB_INCLUDE_DIRS="$LIB_DIR/zlib/include" > $null
  cmake --build . --config release > $null
  cmake --build . --config release --target INSTALL > $null
}

# Install NetCDF
if(!(Test-Path -Path "$LIB_DIR\netcdf") -Or $REBUILD_NETCDF) {
  Write-Host "`nBuilding NetCDF" -ForegroundColor Yellow
  cd $TMP_DIR
  if(Test-Path -Path "$LIB_DIR\netcdf") {
    Remove-Item "$LIB_DIR\netcdf" -Force -Recurse
  }
  mkdir "$LIB_DIR\netcdf" > $null
  if ($ON_APPVEYOR) {
    appveyor DownloadFile $NETCDF_URL -FileName netcdf.zip > $null
  } else {
    Invoke-WebRequest -Uri $NETCDF_URL -OutFile netcdf.zip
  }
  7z x netcdf.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\netcdf-*" "$TMP_DIR\netcdf"
  cd "$TMP_DIR\netcdf"
  mkdir bld > $null
  cd bld
  $LIB_DIR_REV=$LIB_DIR -replace '\\','/'
  cmake .. -G"Visual Studio 15 2017 Win64" -DCMAKE_INSTALL_PREFIX="$LIB_DIR_REV/netcdf" -DCMAKE_BUILD_TYPE=Release -DENABLE_NETCDF_4=ON -DENABLE_DAP=OFF -DUSE_DAP=OFF -DHDF5_DIR="$LIB_DIR_REV/hdf5/cmake" -DHDF5_C_LIBRARY="$LIB_DIR_REV/hdf5/lib/libhdf5.lib" -DHDF5_HL_LIBRARY="$LIB_DIR_REV/hdf5/lib/libhdf5_hl.lib" -DHDF5_INCLUDE_DIR="$LIB_DIR_REV/hdf5/include" -DZLIB_INCLUDE_DIR="$LIB_DIR_REV/zlib/include" -DZLIB_LIBRARY="$LIB_DIR_REV/zlib/lib/zlib.lib" -DCMAKE_INCLUDE_PATH="$LIB_DIR_REV/hdf5/include" > $null
  cmake --build . --config release > $null
  cmake --build . --config release --target INSTALL > $null
}

# Install Gdal
if(!(Test-Path -Path "$LIB_DIR\gdal") -Or $REBUILD_GDAL) {
  Write-Host "`nBuilding Gdal" -ForegroundColor Yellow
  cd $TMP_DIR
  if(Test-Path -Path "$LIB_DIR\gdal") {
    Remove-Item "$LIB_DIR\gdal" -Force -Recurse
  }
  mkdir "$LIB_DIR\gdal" > $null
  if ($ON_APPVEYOR) {
    appveyor DownloadFile $GDAL_URL -FileName gdal.zip > $null
  } else {
    Invoke-WebRequest -Uri $GDAL_URL -OutFile gdal.zip
  }
  7z x gdal.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\gdal-*" "$TMP_DIR\gdal"
  cd "$TMP_DIR\gdal"
  $LIB_DIR_REV=$LIB_DIR -replace '\\','/'
  nmake -f makefile.vc MSVC_VER=$MSC_VER WIN64=1 GDAL_HOME="$LIB_DIR\gdal" CURL_DIR="$LIB_DIR\curl" CURL_INC="-I$LIB_DIR_REV/curl/include" CURL_LIB="$LIB_DIR_REV/curl/lib/libcurl.lib wsock32.lib wldap32.lib winmm.lib" CURL_CFLAGS=-DCURL_STATICLIB > $null
  nmake -f makefile.vc MSVC_VER=$MSC_VER WIN64=1 GDAL_HOME="$LIB_DIR\gdal" CURL_DIR="$LIB_DIR\curl" CURL_INC="-I$LIB_DIR_REV/curl/include" CURL_LIB="$LIB_DIR_REV/curl/lib/libcurl.lib wsock32.lib wldap32.lib winmm.lib" CURL_CFLAGS=-DCURL_STATICLIB install > $null
  nmake -f makefile.vc MSVC_VER=$MSC_VER WIN64=1 GDAL_HOME="$LIB_DIR\gdal" CURL_DIR="$LIB_DIR\curl" CURL_INC="-I$LIB_DIR_REV/curl/include" CURL_LIB="$LIB_DIR_REV/curl/lib/libcurl.lib wsock32.lib wldap32.lib winmm.lib" CURL_CFLAGS=-DCURL_STATICLIB devinstall > $null
}

# Install Jasper
if(!(Test-Path -Path "$LIB_DIR\jasper") -Or $REBUILD_PROJ) {
  Write-Host "`nBuilding Jasper" -ForegroundColor Yellow
  cd $TMP_DIR
  if(Test-Path -Path "$LIB_DIR\jasper") {
    Remove-Item "$LIB_DIR\jasper" -Force -Recurse
  }
  mkdir "$LIB_DIR\jasper" > $null
  if ($ON_APPVEYOR) {
    appveyor DownloadFile $JASPER_URL -FileName jasper.zip > $null
  } else {
    Invoke-WebRequest -Uri $JASPER_URL -OutFile jasper.zip
  }
  7z x jasper.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\jasper-*" "$TMP_DIR\jasper"
  cd "$TMP_DIR\jasper\src\msvc"


msdev jasper.dsp /MAKE "libjasper – Win32 Debug" /REBUILD


  nmake -f Makefile.in INSTDIR="$LIB_DIR\jasper" > $null
  nmake -f makefile.vc INSTDIR="$LIB_DIR\jasper" install-all > $null
}

# Install ecCodes (DOES NOT WORK)
if(!(Test-Path -Path "$LIB_DIR\eccodes") -Or $REBUILD_ECCODES) {
  Write-Host "`nBuilding ecCodes" -ForegroundColor Yellow
  cd $TMP_DIR
  if(Test-Path -Path "$LIB_DIR\eccodes") {
    Remove-Item "$LIB_DIR\eccodes" -Force -Recurse
  }
  mkdir "$LIB_DIR\eccodes" > $null
  if ($ON_APPVEYOR) {
    appveyor DownloadFile $ECCODES_URL -FileName eccodes.tar.gz > $null
  } else {
    Invoke-WebRequest -Uri $ECCODES_URL -OutFile eccodes.tar.gz
  }
  7z x eccodes.tar.gz -o"$TMP_DIR" > $null
  7z x eccodes.tar -o"$TMP_DIR" > $null
  move "$TMP_DIR\eccodes-*" "$TMP_DIR\eccodes"
  cd "$TMP_DIR\eccodes"
  mkdir bld > $null
  cd bld
  cmake .. -G"Visual Studio 15 2017 Win64" -DCMAKE_INSTALL_PREFIX="$LIB_DIR\eccodes" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DDISABLE_OS_CHECK=ON -DENABLE_FORTRAN=OFF -DENABLE_EXTRA_TESTS=OFF > $null
  cmake --build . --config release --target eccodes > $null
}