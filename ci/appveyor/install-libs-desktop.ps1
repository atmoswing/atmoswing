. $PSScriptRoot\libs-common-definitions.ps1

# Options
$CMAKE_URL="https://cmake.org/files/v3.14/cmake-3.14.3-win64-x64.zip"
$TARGET_CPU="x64"
$WX_TARGET_CPU="X64"
$GDAL_WIN64=1

# Setup VS environment
# https://stackoverflow.com/questions/2124753/how-can-i-use-powershell-with-the-visual-studio-command-prompt
pushd "C:\Program Files (x86)\Microsoft Visual Studio\$VS_VER_YR\Community\VC\Auxiliary\Build"
cmd /c "vcvars64.bat&set" |
foreach {
  if ($_ -match "=") {
    $v = $_.split("="); set-item -force -path "ENV:\$($v[0])"  -value "$($v[1])"
  }
}
popd
Write-Host "`nVisual Studio $VS_VER_YR Command Prompt variables set." -ForegroundColor Yellow

set CL=/MP

. $PSScriptRoot\libs-desktop-install.ps1

Get-ChildItem "$LIB_DIR/include"
Get-ChildItem "$LIB_DIR/lib"
