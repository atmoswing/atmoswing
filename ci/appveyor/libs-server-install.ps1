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
  nmake -f makefile.vc BUILD=release MONOLITHIC=0 SHARED=0 USE_OPENGL=0 TARGET_CPU=$WX_TARGET_CPU USE_GUI=0 > $null
  nmake -f makefile.vc BUILD=debug MONOLITHIC=0 SHARED=0 USE_OPENGL=0 TARGET_CPU=$WX_TARGET_CPU USE_GUI=0 > $null
  move "$TMP_DIR\wxwidgets\include" "$LIB_DIR\include"
  copy "$TMP_DIR\wxwidgets\lib\vc_${TARGET_CPU}_lib\mswu\wx\setup.h" "$LIB_DIR\wxwidgets\include\wx\setup.h"
  move "$LIB_DIR\include\wx\msw\rcdefs.h" "$LIB_DIR\include\wx\msw\rcdefs.h_old"
  copy "$TMP_DIR\wxwidgets\lib\vc_${TARGET_CPU}_lib\mswu\wx\msw\rcdefs.h" "$LIB_DIR\include\wx\msw\rcdefs.h"
  move "$TMP_DIR\wxwidgets\lib" "$LIB_DIR\lib"
}
$env:WXWIN = "$LIB_DIR"


. $PSScriptRoot\libs-common-install.ps1


Get-ChildItem "$LIB_DIR/include"
