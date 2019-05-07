
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

if ($stopwatchlibs.Elapsed.TotalMinutes -gt 40) { return }

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

if ($stopwatchlibs.Elapsed.TotalMinutes -gt 40) { return }

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

if ($stopwatchlibs.Elapsed.TotalMinutes -gt 40) { return }

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

if ($stopwatchlibs.Elapsed.TotalMinutes -gt 40) { return }

# Install curl
if(!(Test-Path -Path "$LIB_DIR\include\curl\curl.h") -Or $REBUILD_CURL) {
  Init-Build "curl"
  Download-Lib "curl" $CURL_URL
  7z x curl.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\curl-*" "$TMP_DIR\curl"
  cd "$TMP_DIR\curl\winbuild"
  nmake -f Makefile.vc mode=dll VC=14 DEBUG=NO MACHINE=${TARGET_CPU} > $null
  Copy-Item "$TMP_DIR\curl\builds\libcurl-vc14-${TARGET_CPU}-release-dll-ipv6-sspi-winssl\bin\*" "$LIB_DIR\bin" -force
  Copy-Item "$TMP_DIR\curl\builds\libcurl-vc14-${TARGET_CPU}-release-dll-ipv6-sspi-winssl\include\*" "$LIB_DIR\include" -recurse -force
  Copy-Item "$TMP_DIR\curl\builds\libcurl-vc14-${TARGET_CPU}-release-dll-ipv6-sspi-winssl\lib\*" "$LIB_DIR\lib" -force
}

if ($stopwatchlibs.Elapsed.TotalMinutes -gt 40) { return }

# Install SQLite
if(!(Test-Path -Path "$LIB_DIR\include\sqlite3.h") -Or $REBUILD_SQLITE) {
  Init-Build "sqlite"
  Download-Lib "sqlite_src" $SQLITE_SRC_URL
  Download-Lib "sqlite_dll" $SQLITE_DLL_URL
  Download-Lib "sqlite_tools" $SQLITE_TOOLS_URL
  7z x sqlite_src.zip -o"$TMP_DIR" > $null
  7z x sqlite_dll.zip -o"$TMP_DIR" > $null
  7z x sqlite_tools.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\sqlite-tools*" "$TMP_DIR\sqlitetools"
  move "$TMP_DIR\sqlite-*" "$TMP_DIR\sqlite"
  lib /def:sqlite3.def
  copy "$TMP_DIR\sqlite3.dll" "$LIB_DIR\bin\sqlite3.dll"
  copy "$TMP_DIR\sqlite3.lib" "$LIB_DIR\lib\sqlite3.lib"
  copy "$TMP_DIR\sqlitetools\sqlite3.exe" "$LIB_DIR\bin\sqlite3.exe"
  copy "$TMP_DIR\sqlite\sqlite3.h" "$LIB_DIR\include\sqlite3.h"
  copy "$TMP_DIR\sqlite\sqlite3ext.h" "$LIB_DIR\include\sqlite3ext.h"
}

# Install Proj
if(!(Test-Path -Path "$LIB_DIR\include\proj_api.h") -Or $REBUILD_PROJ) {
  Init-Build "proj"
  Download-Lib "proj" $PROJ_URL
  7z x proj.zip -o"$TMP_DIR" > $null
  move "$TMP_DIR\proj.4-*" "$TMP_DIR\proj"
  cd "$TMP_DIR\proj"
  mkdir build
  cd build
  cmake -G"Visual Studio 15 2017 Win64" -DCMAKE_PREFIX_PATH="$LIB_DIR" -DPROJ_TESTS=OFF -DBUILD_PROJINFO=OFF -DBUILD_CCT=OFF -DBUILD_CS2CS=OFF -DBUILD_GEOD=OFF -DBUILD_GIE=OFF -DBUILD_PROJ=OFF -DBUILD_PROJINFO=OFF -DBUILD_LIBPROJ_SHARED=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$LIB_DIR" .. > $null
  cmake --build . --config Release > $null
  cmake --build . --config Release --target INSTALL > $null
  copy "$LIB_DIR\bin\proj*.dll" "$LIB_DIR\bin\proj.dll"
}

if ($stopwatchlibs.Elapsed.TotalMinutes -gt 40) { return }

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

if ($stopwatchlibs.Elapsed.TotalMinutes -gt 40) { return }

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
  cmake .. -G"$VS_VER" -DCMAKE_INSTALL_PREFIX="$LIB_DIR_REV" -DCMAKE_BUILD_TYPE=Release -DENABLE_NETCDF_4=ON -DENABLE_DAP=OFF -DUSE_DAP=OFF -DBUILD_UTILITIES=OFF -DENABLE_TESTS=OFF -DHDF5_DIR="$LIB_DIR_REV/cmake" -DHDF5_C_LIBRARY="$LIB_DIR_REV/lib/libhdf5.lib" -DHDF5_HL_LIBRARY="$LIB_DIR_REV/lib/libhdf5_hl.lib" -DHDF5_INCLUDE_DIR="$LIB_DIR_REV/include" -DZLIB_INCLUDE_DIR="$LIB_DIR_REV/include" -DZLIB_LIBRARY="$LIB_DIR_REV/lib/zlib.lib" -DCMAKE_INCLUDE_PATH="$LIB_DIR_REV/include" > $null
  cmake --build . --config release > $null
  cmake --build . --config release --target INSTALL > $null
}

# Install ecCodes
if(!(Test-Path -Path "$LIB_DIR\include\eccodes.h") -Or $REBUILD_ECCODES) {
  Init-Build "eccodes"
  Write-Host "`nDownloading eccodes from $ECCODES_URL" -ForegroundColor Yellow
  Invoke-WebRequest -Uri $ECCODES_URL -OutFile "eccodes.tar.gz"
  7z x eccodes.tar.gz -o"$TMP_DIR" > $null
  7z x eccodes.tar -o"$TMP_DIR" > $null
  move "$TMP_DIR\eccodes-*" "$TMP_DIR\eccodes"
  cd "$TMP_DIR\eccodes"
  mkdir bld > $null
  cd bld
  Copy-Item "$PATCH_DIR\grib_lex.c" -Destination "$TMP_DIR\eccodes\src\grib_lex.c"
  cmake .. -G"$VS_VER" -DCMAKE_INSTALL_PREFIX="$LIB_DIR" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DENABLE_JPG=ON -DENABLE_PYTHON=OFF -DENABLE_FORTRAN=OFF -DENABLE_ECCODES_THREADS=OFF -DCMAKE_PREFIX_PATH="$LIB_DIR" -DDISABLE_OS_CHECK=ON > $null
  cmake --build . --config release --target libs > $null
  copy "$TMP_DIR\eccodes\bld\lib\Release\eccodes.lib" "$LIB_DIR\lib\eccodes.lib"
  copy "$TMP_DIR\eccodes\src\*.h" "$LIB_DIR\include\"
  copy "$TMP_DIR\eccodes\bld\src\eccodes_version.h" "$LIB_DIR\include\"
  Copy-Item "$TMP_DIR\eccodes\definitions" -Destination "$LIB_DIR\share\eccodes\definitions" -Recurse
}