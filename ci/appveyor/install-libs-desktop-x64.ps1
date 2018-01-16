. .\libs-common-definitions.ps1

# Options
$VS_VER="Visual Studio 15 2017 Win64"
$CMAKE_URL="https://cmake.org/files/v3.10/cmake-3.10.0-win64-x64.zip"
$TARGET_CPU="x64"
$WX_TARGET_CPU="X64"
$GDAL_WIN64=1

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

. .\libs-common-install.ps1

Get-ChildItem "$LIB_DIR/include"
